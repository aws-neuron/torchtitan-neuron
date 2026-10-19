# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Post-optimizer MXFP8 weight quantization for SimpleFSDP.

The stock MXFP8 path all-gathers a bf16 weight and quantizes it inside the GEMM every
step. This path quantizes each shard once after ``optimizer.step()``, so the forward
all-gathers 1-byte fp8 plus its e8m0 scale instead of bf16.

Opt in through the config: attach ``NeuronMXFP8LinearConverter`` with
``prequantize_weights=True`` in a recipe. The converter swaps each ``MXFP8Linear`` for a
``NeuronMXFP8Linear`` whose ``forward`` consumes pre-quantized encodings and whose
``refresh_weight_derived_state`` re-quantizes the local shard. The trainer calls that
method (duck-typed) after init_weights, after checkpoint load, and after each optimizer
step -- the analogue of ``precompute_float8_dynamic_scale_for_fsdp``.

Two encodings are needed because MX block scaling runs along the *contraction* axis,
which differs between the two GEMMs that consume the weight:

    forward  out    = x @ W.T        contracts in_features   -> blocks along in
    dgrad    grad_x = grad_out @ W   contracts out_features  -> blocks along out

so the fprop encoding comes from ``W`` and the dgrad one from ``W.t()``, both blocked on
the last dim. The transpose has to happen eagerly, here: an in-graph ``.t()`` on a
pre-quantized operand loses the packing the compiler requires.

Both encodings live on ``_PrequantWeightOperands``; see it for their shapes. Each shard
dim plays both roles across the two encodings, so both dims must satisfy both kernel
bounds -- see ``MIN_K_BY_SCHEME`` / ``MIN_F``, enforced by ``_check_shard`` on the first
refresh. Both bounds are met only by the dim-1-FSDP-sharded dev flavors, which require
SimpleFSDP dim-1 sharding support.

``prequant_scheme`` picks the packed layout: ``wrapX`` (nkilib's own quantize kernel) or
``1x32``. They differ only in which K occupies which packed slot, so the same value must
reach the quantize kernel and the matmul.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import torch
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.components.quantization.mx import MXFP8Linear, MXFP8LinearConverter

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn

# nkilib QuantScheme values, spelled as matmul_mxfp8's ``quant_scheme`` expects them.
SCHEME_WRAPX = "wrapX"
SCHEME_1X32 = "1x32"
_SCHEMES = (SCHEME_WRAPX, SCHEME_1X32)

# nkilib quantize_block_mxfp8_kernel bounds: MIN_K_FOR_QUANTIZATION (128) on the
# contraction dim, MIN_F_FOR_QUANTIZATION on the free dim.
#
# The floor on K is the KERNEL's 128 under 1x32 and 512 under wrapX, and the difference
# is about ordering, not the kernel. A gathered encoding is per-rank encodings
# concatenated along packed dim 0. Under 1x32 that axis is K-consecutive, so any chunk
# boundary is a K boundary and only the packing's /4 and the 32-element block matter.
# Under wrapX the kernel interleaves each 512-element K tile with stride 128, so per-rank
# tilings coincide with the full-K tiling only when each chunk is a whole 512-tile; a
# sub-512 chunk uses strides that never occur in the full-K decomposition and the matmul
# then pairs mismatched K silently (~100% error, no exception).
MIN_K_BY_SCHEME = {SCHEME_WRAPX: 512, SCHEME_1X32: 128}
MIN_F = 8

__all__ = ["NeuronMXFP8Linear", "NeuronMXFP8LinearConverter"]

# Instance attributes the refresh writes and the forward reads.
_OPERANDS = "_mxfp8_operands"
_SPEC = "_mxfp8_mesh_spec"


def _canonical_scheme(scheme: str) -> str:
    """Validate a scheme name and return its canonical spelling.

    Spelling matters: it is passed through to nkilib, which compares against its own
    QuantScheme values, so accept case-insensitively but hand on the canonical form.
    """
    for known in _SCHEMES:
        if scheme.lower() == known.lower():
            return known
    raise ValueError(
        f"prequant_scheme={scheme!r} is not a known scheme; expected one of {_SCHEMES}"
    )


@dataclass(frozen=True, slots=True)
class _PrequantWeightOperands:
    """The packed MXFP8 encodings derived from one local weight shard.

    Named and shaped after upstream's ``_MXFP8LinearOperands``
    (``torchtitan/components/quantization/mxfp8/tensor.py``, pytorch/torchtitan#4203) so
    adopting that overhaul is a rename rather than a rewrite. It cannot share upstream's
    field *set*: there, square 32x32 scale tiles are transpose-invariant, so one qdata
    allocation serves both GEMMs and only the scales differ. Neuron's quantize kernels
    offer no 2D scheme -- nkilib ``QuantScheme`` is ``WRAPX | _1x32``, both blocked along
    K alone -- so fprop and dgrad are independently quantized and each owns its qdata.

    For a local shard ``[O, I]``, contraction dim leading the packed layout:

        fprop (contracts I)   qdata [I/4, O*4]   scale [I/4, O]
        dgrad (contracts O)   qdata [O/4, I*4]   scale [O/4, I]

    Fields are distinct allocations, which is what lets the refresh refill them in
    place. Field order is the gather order in ``_prequant_forward``.
    """

    qdata_fprop: torch.Tensor
    scale_fprop: torch.Tensor
    qdata_dgrad: torch.Tensor
    scale_dgrad: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))


def _quantize_mx(t: torch.Tensor, scheme: str):
    """Quantize to packed MXFP8, contracting over ``t``'s LAST dim.

    Returns ``(data [K/4, F*4], scales [K/4, F])`` for ``t`` of shape ``[F, K]`` -- the
    swizzled x4-packed layout ``matmul_mxfp8`` requires of a pre-quantized operand. Note
    the contraction dim leads the packed result, which is why the gather placements below
    invert relative to the weight's own dims.
    """
    if t.dtype not in (torch.bfloat16, torch.float16):
        t = t.to(torch.bfloat16)
    if not t.is_contiguous():
        t = t.contiguous()
    return torch.ops.neuron.quantize_mx_nki(t, FP8_DTYPE, scheme)


def _build_operands(
    shard: torch.Tensor,
    scheme: str,
    out: _PrequantWeightOperands | None = None,
) -> _PrequantWeightOperands:
    """Quantize a local weight shard into both encodings.

    Allocates a new operands when ``out`` is None, otherwise refills ``out``'s existing
    tensors in place and returns it. Mirrors upstream's ``_build_operands`` contract.

    The dgrad encoding materialises ``[I, O]`` here, eagerly: an in-graph ``.t()`` on a
    pre-quantized operand loses the packing the compiler requires.
    """
    fresh = _PrequantWeightOperands(
        *_quantize_mx(shard, scheme), *_quantize_mx(shard.t(), scheme)
    )
    if out is None:
        return fresh
    for existing, value in zip(out.tensors(), fresh.tensors(), strict=True):
        # A raise, not an assert: this guards against silent corruption rather than a
        # programmer typo, and asserts vanish under -O. copy_ broadcasts, so a mismatch
        # slipping through would write a wrong-but-plausible encoding instead of failing.
        if existing.shape != value.shape or existing.dtype != value.dtype:
            raise RuntimeError(
                f"encoding changed shape/dtype across refresh: "
                f"{tuple(existing.shape)}/{existing.dtype} -> "
                f"{tuple(value.shape)}/{value.dtype}"
            )
        existing.copy_(value)
    return out


def _check_shard(name: str, o: int, i: int, scheme: str) -> None:
    """Validate a local weight shard against the NKI quantize kernel's constraints.

    The kernel needs its contraction dim (``K``) a multiple of the active scheme's
    ``MIN_K`` and its free dim (``F``) a multiple of ``MIN_F``. Both encodings are taken
    from the same shard with the roles swapped, so each dim has to satisfy both bounds.
    """
    min_k = MIN_K_BY_SCHEME[scheme]
    for dim_name, k, f in (("fprop", i, o), ("dgrad", o, i)):
        if k % min_k:
            raise ValueError(
                f"{name}: {dim_name} encoding contracts over {k}, which must be a "
                f"multiple of {min_k} under scheme {scheme}; got {k} % {min_k} = "
                f"{k % min_k}. Either a block straddles two ranks (so the local amax is "
                f"not the block amax) or the per-rank K tiling does not line up with the "
                f"full-K tiling. Shard the other axis, reduce the FSDP degree, widen the "
                f"model dim, or -- if on wrapX -- try the 1x32 scheme, whose floor is "
                f"{MIN_K_BY_SCHEME[SCHEME_1X32]}."
            )
        if f % MIN_F:
            raise ValueError(
                f"{name}: {dim_name} encoding's free dim ({f}) must be a multiple of "
                f"{MIN_F}; got {f} % {MIN_F} = {f % MIN_F}."
            )


class _PrequantMM(torch.autograd.Function):
    """``x @ W.T`` from pre-quantized weight encodings.

    Only the weight arrives pre-quantized; activations change every step, so ``fprop`` and
    ``dgrad`` quantize theirs in-kernel and ``wgrad`` quantizes both operands inline.

    ``weight`` is passed but never read -- the encodings carry the values. It exists
    purely as an autograd edge back to the parameter, since the encodings are plain fp8
    tensors with no grad path; without it ``grad_w`` is silently discarded and every
    linear stays frozen. Must be the RAW sharded parameter, not ``mod.weight``: that
    property runs the SimpleFSDP parametrization and re-introduces the bf16 all-gather
    this path exists to avoid.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        qdata_fprop,
        scale_fprop,
        qdata_dgrad,
        scale_dgrad,
        wspec,
        scheme,
    ):
        out = torch.ops.neuron.prequantized_mx_scaled_mm_nki(
            x, qdata_fprop, scale_fprop, x.dtype, FP8_DTYPE, scheme
        )
        ctx.scheme = scheme
        ctx.save_for_backward(x, qdata_dgrad, scale_dgrad)
        ctx.wspec = wspec
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, qdata_dgrad, scale_dgrad = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        # grad_x = grad_out @ W : contract out_features, so rhs is the [I, O] encoding.
        grad_x = torch.ops.neuron.prequantized_mx_scaled_mm_nki(
            grad_out,
            qdata_dgrad,
            scale_dgrad,
            grad_out.dtype,
            FP8_DTYPE,
            ctx.scheme,
        )

        # grad_W = grad_out.T @ x : both operands are activations, so the fully fused op
        # quantizes them inline. It computes lhs @ rhs.T, so rhs is x.T.
        grad_w = torch.ops.neuron.fused_quantize_mx_scaled_mm_nki(
            grad_out.t().contiguous(),
            x.t().contiguous(),
            grad_out.dtype,
            FP8_DTYPE,
        )
        # Positions: x, weight, qdata_fprop, scale_fprop, qdata_dgrad, scale_dgrad,
        # wspec, scheme
        return (
            grad_x,
            _to_param_layout(grad_w, ctx.wspec),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _to_param_layout(grad_w, wspec):
    """Reshape ``grad_w`` from the activation layout into the parameter's own sharding.

    Each DP rank saw a different batch slice, so its ``grad_w`` is one addend of the true
    gradient. Declaring the DP axis ``Partial`` and redistributing to the parameter's
    placements is exactly the reduce-scatter SimpleFSDP's parametrization backward would
    have done, so this costs no more than the non-prequant path.

    TP placements are read off ``grad_w``, not copied from the parameter: DTensor may have
    satisfied the GEMM by redistributing ``x`` to ``Replicate`` on TP, leaving ``grad_w``
    TP-whole, and asserting the parameter's ``Shard`` would then overstate the global
    shape by the TP degree (autograd rejects the gradient's shape).
    """
    if wspec is None:
        return grad_w
    mesh, dp_i, raw_placements = wspec
    if isinstance(grad_w, DTensor):
        local = grad_w.to_local()
        tp_placements = list(grad_w.placements)
    else:
        local = grad_w
        tp_placements = [Replicate()] * (mesh.ndim - 1)

    src, tp_at = [], 0
    for i in range(mesh.ndim):
        if i == dp_i:
            src.append(Partial())
        else:
            src.append(
                tp_placements[tp_at] if tp_at < len(tp_placements) else Replicate()
            )
            tp_at += 1

    full = DTensor.from_local(local, mesh, src, run_check=False)
    return full.redistribute(placements=list(raw_placements))


_DP_MESH_NAMES = ("dp_shard_cp", "dp_shard", "fsdp", "dp")


def _retarget(p, new_dim):
    """Same placement type, rebased onto ``new_dim``.

    ``_StridedShard``'s ``split_factor`` must be carried over: when TP has already
    split a dim, FSDP's split of that same dim is interleaved, and reassembling it as
    a plain ``Shard`` would concatenate the pieces in the wrong order -- silently
    wrong weights rather than an error.
    """
    if isinstance(p, _StridedShard):
        return _StridedShard(new_dim, split_factor=p.split_factor)
    if isinstance(p, Shard):
        return Shard(new_dim)
    return p


def _placements(raw_placements, transposed: bool):
    out = []
    for p in raw_placements:
        d = getattr(p, "dim", None)
        if d in (0, 1) and transposed:
            out.append(_retarget(p, 1 - d))
        else:
            out.append(p)
    return list(out)


def _build_spec(raw):
    """Gather plan for one weight, from the RAW parameter (not the property).

    ``mod.weight`` runs SimpleFSDP's parametrization and returns the already-gathered
    tensor on the TP-only mesh. ``mod._parameters["weight"]`` is the only place the
    ``fsdp`` mesh axis and its ``_StridedShard`` placement are visible.

    Placements are in PACKED coordinates, contraction dim leading: the fprop encoding
    contracts in_features, so weight dim 1 becomes packed dim 0 (the ``transposed``
    mapping); the dgrad encoding contracts out_features, so dims carry over unchanged.
    Chunk sizes stay integral under the packing's /4 and *4 because ``_check_shard``
    already required each local dim to be a multiple of the scheme's ``MIN_K`` and
    ``MIN_F``.

    Returns (mesh, dp_index, fprop_placements, dgrad_placements, tp_mesh, tp_fprop, tp_dgrad)
    or None when there is no DP axis to gather.
    """
    if not isinstance(raw, DTensor):
        return None
    mesh = raw.device_mesh
    names = list(mesh.mesh_dim_names or [])
    dp_i = next((i for i, n in enumerate(names) if n in _DP_MESH_NAMES), None)
    if dp_i is None:
        return None
    tp_names = tuple(n for i, n in enumerate(names) if i != dp_i)
    tp_mesh = mesh[tp_names] if tp_names else None
    tp_idx = [i for i in range(len(names)) if i != dp_i]
    tp_raw = [raw.placements[i] for i in tp_idx]
    return (
        mesh,
        dp_i,
        _placements(raw.placements, True),
        _placements(raw.placements, False),
        tp_mesh,
        _placements(tp_raw, True),
        _placements(tp_raw, False),
    )


def _gather(local, spec, transposed: bool):
    """All-gather one local encoding over the DP axis only, inside the traced forward.

    Mirrors ``ReplicateComputation.replicate_compute``: redistribute the DP axis to
    Replicate leaving TP alone, then re-wrap on the TP mesh so the result is a DTensor on
    the same mesh as the activation (a plain tensor would mix Tensor and DTensor and raise).

    fp8 is not a supported collective dtype, so the payload is sent as ``uint8``.
    Both float8_e4m3fn and float8_e8m0fnu are 1 byte, so the view is
    shape-preserving and the placements stay valid.
    """
    if spec is None:
        return local
    mesh, dp_i, pl_fprop, pl_dgrad, tp_mesh, tp_fprop, tp_dgrad = spec
    pl = list(pl_dgrad if transposed else pl_fprop)

    # Capture the true element dtype BEFORE the byte view: the receiver has to know
    # what the bytes were, and guessing wrong is silently wrong rather than an error.
    elem_dtype = local.dtype
    payload = local if elem_dtype == torch.uint8 else local.view(torch.uint8)

    d = DTensor.from_local(payload, mesh, pl, run_check=False)
    target = list(pl)
    # DTensor issues all gather on DP axis and no other
    target[dp_i] = Replicate()
    gathered = d.redistribute(placements=target).to_local()
    if elem_dtype != torch.uint8:
        gathered = gathered.view(elem_dtype)
    if tp_mesh is None:
        return gathered
    return DTensor.from_local(
        gathered, tp_mesh, tp_dgrad if transposed else tp_fprop, run_check=False
    )


def _prequant_forward(self, input: torch.Tensor) -> torch.Tensor:
    """``MXFP8Linear.forward`` that consumes pre-quantized weights."""
    operands = getattr(self, _OPERANDS, None)
    if operands is None:
        raise RuntimeError(
            "MXFP8 pre-quantized weights are missing. refresh_weight_derived_state() "
            "must run once after init_weights and before the first forward."
        )
    orig_shape = input.shape
    x = input.reshape(-1, orig_shape[-1])
    spec = getattr(self, _SPEC)
    # The RAW sharded parameter, not self.weight: reading the property would run the
    # parametrization and all-gather the bf16 weight. Passed only as a grad edge.
    raw = self._parameters.get("weight")
    wspec = None
    if spec is not None and isinstance(raw, DTensor):
        wspec = (spec[0], spec[1], tuple(raw.placements))

    # The encodings cross the autograd boundary as separate positional tensors, not as
    # the operands object: tensors reached through a dataclass argument are not graph
    # inputs to a traced autograd.Function, so the collectives producing them would fall
    # outside the graph. The grouping is for the module state and the quantizer.
    out = _PrequantMM.apply(
        x,
        raw,
        _gather(operands.qdata_fprop, spec, False),
        _gather(operands.scale_fprop, spec, False),
        _gather(operands.qdata_dgrad, spec, True),
        _gather(operands.scale_dgrad, spec, True),
        wspec,
        self._prequant_scheme,
    )
    out = out.reshape(*orig_shape[:-1], out.shape[-1])
    if self.bias is not None:
        out = out + self.bias.to(out.dtype)
    return out


if MXFP8Linear is not None:

    class NeuronMXFP8Linear(MXFP8Linear):
        """MXFP8Linear that pre-quantizes its weight shard once per optimizer step.

        ``forward`` consumes the packed encodings; ``refresh_weight_derived_state``
        rebuilds them from the current shard and is called by the trainer after
        init_weights, after checkpoint load, and after each optimizer step.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(MXFP8Linear.Config):
            prequant_scheme: str = SCHEME_WRAPX

        def __init__(self, config: Config):
            super().__init__(config)
            self._prequant_scheme = _canonical_scheme(config.prequant_scheme)
            setattr(self, _OPERANDS, None)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return _prequant_forward(self, input)

        @torch.no_grad()
        def refresh_weight_derived_state(self) -> None:
            """Re-quantize this rank's weight shard (fprop + dgrad encodings).

            Reads ``_parameters["weight"]``, NOT ``self.weight``: the latter is a property
            that runs SimpleFSDP's parametrization and returns the already-gathered weight.
            We want the local shard -- quantize 1/FSDP of the weight and all-gather 1 byte
            per element instead of 2.
            """
            raw = self._parameters.get("weight")
            local = raw._local_tensor if isinstance(raw, DTensor) else raw
            if local is None or local.ndim != 2:
                return
            inner = getattr(local, "_tensor", local)

            # Mesh layout is fixed for the run, so resolve the gather plan once, and
            # validate the local shard against the kernel bounds on that first pass.
            if not hasattr(self, _SPEC):
                _check_shard(
                    type(self).__name__,
                    local.shape[0],
                    local.shape[1],
                    self._prequant_scheme,
                )
                setattr(self, _SPEC, _build_spec(raw))

            # Refill IN PLACE. Rebinding would hand the module new tensor objects each step
            # while the compiled graph still holds the ones it captured on the first trace,
            # so the linears would keep reading step-0 encodings. No version-counter guard
            # is needed: refresh runs after backward has consumed them.
            operands = getattr(self, _OPERANDS)
            refilled = _build_operands(inner, self._prequant_scheme, out=operands)
            if operands is None:
                setattr(self, _OPERANDS, refilled)

    class NeuronMXFP8LinearConverter(MXFP8LinearConverter):
        """MXFP8 Linear converter with an opt-in pre-quantized SimpleFSDP weight path."""

        @dataclass(kw_only=True, slots=True)
        class Config(MXFP8LinearConverter.Config):
            prequantize_weights: bool = False
            prequant_scheme: str = SCHEME_WRAPX

        def __init__(self, config: Config):
            super().__init__(config)
            if config.prequantize_weights:
                _canonical_scheme(config.prequant_scheme)
                if config.kernel_preference != "neuron":
                    raise ValueError(
                        "prequantize_weights=True requires kernel_preference='neuron'"
                    )

        def convert(self, model_config):
            model_config = super().convert(model_config)
            if not self.config.prequantize_weights:
                return model_config
            for _fqn, cfg, parent, attr in model_config.traverse(MXFP8Linear.Config):
                new_config = NeuronMXFP8Linear.Config(
                    **{f.name: getattr(cfg, f.name) for f in fields(cfg)},
                    prequant_scheme=self.config.prequant_scheme,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)
            logger.info(
                "MXFP8 pre-quantized SimpleFSDP weight path enabled (scheme=%s)",
                self.config.prequant_scheme,
            )
            return model_config

else:
    NeuronMXFP8Linear = None
    NeuronMXFP8LinearConverter = None
