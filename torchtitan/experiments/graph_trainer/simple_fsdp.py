# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count

import torch
import torch.nn as nn

from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from torchtitan.protocols.module import Module

_active_parametrization = True


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


def _distribute_dtensor(
    tensor: DTensor,
    device_mesh: DeviceMesh,
    dp_placements: Sequence[Placement],
) -> DTensor:
    """
    Below are experimental enhancements to distribute a DTensor.
    This helps enable Simple FSDP + TP/EP, in which
        inner spec/mesh is TP/EP spec/mesh
        outer spec/mesh is FSDP/DDP/HSDP spec/mesh
    """
    inner_spec = tensor._spec
    outer_mesh, inner_mesh = device_mesh, inner_spec.mesh
    spanned_mesh = DeviceMesh._concatenate([outer_mesh, inner_mesh])

    if len(dp_placements) == 1:
        assert dp_placements[0].is_replicate() or dp_placements[0].is_shard()
        if dp_placements[0].is_shard():
            # For FSDP + EP/TP/EP+TP
            assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
            shard_dim = dp_placements[0].dim
            split_factor = inner_spec.num_shards_map[shard_dim]
            tensor_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else dp_placements[0]
                ),
            ) + inner_spec.placements
        else:
            # For DDP + TP/EP
            assert len(inner_spec.placements) == 1
            tensor_placement = (dp_placements[0], inner_spec.placements[0])
    elif len(dp_placements) == 2:
        # HSDP + EP/TP/EP+TP. The two DP placements are one Shard and one
        # Replicate; their order follows the DP mesh axis order, which is
        # replicate-first for the default ``dense`` mesh and fsdp-first for the
        # ``dense_fsdp_first`` mesh that only the llama3 SimpleFSDP path opts
        # into. Handle either order by locating the Shard rather than assuming
        # a fixed position.
        assert (
            sum(p.is_shard() for p in dp_placements) == 1
            and sum(p.is_replicate() for p in dp_placements) == 1
        ), f"HSDP expects one Shard and one Replicate placement, got {dp_placements}"
        assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
        shard_idx = 0 if dp_placements[0].is_shard() else 1
        shard_dim = dp_placements[shard_idx].dim
        split_factor = inner_spec.num_shards_map[shard_dim]
        dp_tensor_placement = list(dp_placements)
        if split_factor > 1:
            dp_tensor_placement[shard_idx] = _StridedShard(
                shard_dim, split_factor=split_factor
            )
        tensor_placement = tuple(dp_tensor_placement) + inner_spec.placements
    else:
        raise ValueError(
            f"Unsupported placement {dp_placements} for distributing DTensor {tensor}"
        )

    # HSDP case needs 2 placements for 2D outer_mesh
    current_placements = (Replicate(),) * len(dp_placements)
    target_placements = tuple(dp_placements)
    # The DP mesh distributes the TP/EP-local tensor, not the global tensor.
    local_meta = TensorMeta(
        tensor._local_tensor.shape, tensor._local_tensor.stride(), tensor.dtype
    )

    current_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=current_placements,
        tensor_meta=local_meta,
    )
    target_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=target_placements,
        tensor_meta=local_meta,
    )
    result_tensor = redistribute_local_tensor(
        tensor._local_tensor,
        current_spec=current_spec,
        target_spec=target_spec,
    )
    return DTensor(
        result_tensor.requires_grad_(tensor.requires_grad),
        DTensorSpec(
            mesh=spanned_mesh,
            placements=tensor_placement,
            tensor_meta=inner_spec.tensor_meta,
        ),
        requires_grad=tensor.requires_grad,
    )


# Cache of (original_class, param_names) -> wrapper class, so all instances
# of the same module type share one SimpleFSDP class for torch.compile reuse.
_wrap_class_cache: dict[tuple[type, frozenset[str]], type] = {}
# Suffixes each wrapper class name, as upstream does: one module type can need
# several wrappers (nn.Linear with and without a bias), and pickle resolves each
# by name. The id depends only on wrap order, so it matches across processes.
_wrap_class_id = count()


def _register_parametrization(
    module: nn.Module, param_names: list[str], parametrization: nn.Module
) -> None:
    """
    It works with state_dict without incurring parametrization calls because
    state_dict accesses parameters directly from self._parameters, not from getters
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L2141
    TODO: In checkpoint saving/loading, avoid parametrization calls when calling
    get_model_state_dict func in torchtitan's torchtitan/components/checkpoint.py.
    """
    param_name_to_property = {
        param_name: property(
            lambda self, pn=param_name: parametrization(self._parameters[pn])
        )
        for param_name in param_names
    }
    cache_key = (module.__class__, frozenset(param_names))
    if cache_key in _wrap_class_cache:
        module_cls = _wrap_class_cache[cache_key]
    else:
        module_cls = type(
            f"SimpleFSDP{module.__class__.__name__}_{next(_wrap_class_id)}",
            (module.__class__,),
            param_name_to_property,
        )
        # Expose the dynamically created class as a real, importable symbol
        # so that pickle/GraphPickler can resolve it during serialization.
        sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls
        _wrap_class_cache[cache_key] = module_cls
    module.__class__ = module_cls


class ReplicateComputation(Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mode: str,
        mp_policy: MixedPrecisionPolicy | None,
        full_dtensor: bool = False,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.mode = mode
        self.compute_placements: list[Placement] = [Replicate()] * self.device_mesh.ndim
        self.grad_placements: list[Placement] = [
            Partial(reduce_op="sum")
        ] * self.device_mesh.ndim
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype
        self.full_dtensor = full_dtensor

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)
        # support FSDP/DDP/HSDP + EP + TP (assuming TP shards the inner-most dim)
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
        if non_dp_mesh_dims > 0:
            if self.full_dtensor:
                raise NotImplementedError(
                    "full_dtensor not implemented for nD parallelisms"
                )
            dp_mesh = self.device_mesh
            non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
            non_dp_mesh_dim_names = tuple(
                x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:]
            )
            non_dp_mesh = x._spec.mesh[non_dp_mesh_dim_names]
            # Preserve the true TP/EP-local shape when DP shards are uneven.
            dp_shape, _ = compute_local_shape_and_global_offset(
                x.shape, non_dp_mesh, non_dp_placements, skip_offset=True
            )
            dp_stride = compute_local_stride(x.stride(), dp_shape)
            # re-wrap 2D DTensor to 1D DTensor on dp_mesh for efficient FSDP all-gather
            sharded_local_tensor = x.to_local()
            sharded_dtensor = DTensor.from_local(
                sharded_local_tensor,
                dp_mesh,
                self.param_sharding,
                shape=torch.Size(dp_shape),
                stride=dp_stride,
            )

            # the actual FSDP's fwd all-gather & bwd reduce-scatter
            # DDP's bwd all-reduce on dp_mesh
            replicated_dtensor = sharded_dtensor.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )

            # re-wrap all-gathered DTensor on dp_mesh to be on non_dp_mesh
            # TODO: DTensor should support this mesh collapsing operation
            replicated_local_tensor = replicated_dtensor.to_local(
                grad_placements=self.grad_placements
            )

            output = DTensor.from_local(
                replicated_local_tensor,
                non_dp_mesh,
                non_dp_placements,
                shape=x.shape,
                stride=x.stride(),
            )
        elif non_dp_mesh_dims == 0:
            output = x.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )

            if not self.full_dtensor:
                output = output.to_local(grad_placements=self.grad_placements)
        else:
            raise AssertionError(
                f"Unsupported replicate compute on placement {x._spec.placements} for DTensor {x}"
            )

        return output

    def forward(self, x: DTensor) -> torch.Tensor:
        global _active_parametrization
        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_active_parametrization():
        #     model.init_states()
        if not _active_parametrization:
            return x

        output = self.replicate_compute(x)
        return output


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    full_dtensor: bool = False,
) -> nn.Module:
    param_sharding: tuple[Placement, ...]
    if mode == "replicate":
        param_sharding = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(shard_dim),)
    elif mode == "hybrid_shard":
        assert (
            device_mesh.ndim == 2
        ), "hybrid sharded data parallel requires 2D DeviceMesh"
        # Place Replicate on the dp_replicate axis and Shard on the shard axis
        # (fsdp/efsdp) by name, so the placement is correct regardless of the DP
        # mesh axis order: replicate-first for the default ``dense`` mesh, and
        # fsdp-first for the ``dense_fsdp_first`` mesh that only the llama3
        # SimpleFSDP path opts into. Deriving from names avoids coupling the
        # placement to a fixed axis order (which would silently invert sharding
        # for a differently-ordered mesh, e.g. the MoE experts EDP mesh).
        mesh_axis_names = device_mesh.mesh_dim_names
        assert (
            mesh_axis_names is not None and "dp_replicate" in mesh_axis_names
        ), f"hybrid_shard expects a named 2D DP mesh with a dp_replicate axis, got {mesh_axis_names}"
        param_sharding = tuple(
            Replicate() if name == "dp_replicate" else Shard(shard_dim)
            for name in mesh_axis_names
        )
    else:
        raise ValueError(f"Unsupported mode {mode}")

    modules = list(model.modules())

    for mod in modules:
        params_dict = dict(mod.named_parameters(recurse=False))
        # we shouldn't apply data parallel to the modules that are already
        # sharded by data parallel
        if "SimpleFSDP" in mod.__class__.__name__:
            continue

        for p_name, p in params_dict.items():
            if p is not None and p.numel() > 0:
                distribute_tensor_func = (
                    _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
                )
                mod.register_parameter(
                    p_name,
                    nn.Parameter(
                        distribute_tensor_func(p, device_mesh, param_sharding),
                        requires_grad=p.requires_grad,
                    ),
                )

                # to be compatible with DCP, we use a customized _register_parametrization
                # instead of nn.utils.parametrize.register_parametrization here
                # nn.utils.parametrize.register_parametrization(
                #     mod,
                #     p_name,
                #     ReplicateComputation(
                #         device_mesh,
                #         param_sharding,
                #         mode,
                #         mp_policy=mp_policy,
                #     ),
                #     unsafe=True,
                # )

        _register_parametrization(
            mod,
            list(params_dict.keys()),
            ReplicateComputation(
                device_mesh,
                param_sharding,
                mode,
                mp_policy=mp_policy,
                full_dtensor=full_dtensor,
            ),
        )
    return model
