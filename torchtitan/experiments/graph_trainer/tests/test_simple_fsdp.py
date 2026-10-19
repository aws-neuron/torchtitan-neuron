# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

import pickle
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.graph_trainer import simple_fsdp
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.models.common.decoder import Decoder, TransformerBlock


class TestGraphTrainerQwen3JitActivationCheckpoint(unittest.TestCase):
    def test_qwen3_jit_full_ac_wraps_layers(self):
        from torchtitan.config import ParallelismConfig, TrainingConfig
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.qwen3 import parallelize

        class FakeParallelDims:
            seq_len_divisor = 1
            cp_enabled = False
            tp_enabled = False
            ep_enabled = False

        class TinyQwen3LikeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2)])

        model = TinyQwen3LikeModel()
        original_layer = model.layers[0]
        with (
            patch.object(parallelize, "annotate_qwen3"),
            patch.object(
                parallelize, "apply_simple_fsdp", lambda module, **kwargs: module
            ),
            patch.object(parallelize, "apply_compile", lambda module, **kwargs: module),
            patch.object(parallelize, "maybe_apply_ep_overlap_eager_chunking"),
        ):
            result = parallelize.parallelize_qwen3(
                model,
                parallel_dims=FakeParallelDims(),
                training=TrainingConfig(seq_len=1),
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(mode="jit"),
                ac_config=FullAC.Config(),
                dump_folder="dump",
            )

        self.assertIs(result, model)
        self.assertIsNot(model.layers[0], original_layer)
        self.assertTrue(hasattr(model.layers[0], "_checkpoint_wrapped_module"))
        self.assertIs(model.layers[0]._checkpoint_wrapped_module, original_layer)


class TestGraphTrainerDeepSeekV3JitActivationCheckpoint(unittest.TestCase):
    def test_deepseek_v3_jit_full_ac_wraps_layers(self):
        from torchtitan.config import ParallelismConfig, TrainingConfig
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.deepseek_v3 import parallelize

        class FakeParallelDims:
            seq_len_divisor = 1
            cp_enabled = False
            tp_enabled = False
            ep_enabled = False

        class TinyDeepSeekV3LikeModel(nn.Module):
            class Config:
                layers = []

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2)])
                self.config = self.Config()

        model = TinyDeepSeekV3LikeModel()
        original_layer = model.layers[0]

        def assert_ac_applied_before_simple_fsdp(module, **_kwargs):
            self.assertTrue(hasattr(module.layers[0], "_checkpoint_wrapped_module"))
            return module

        with (
            patch.object(parallelize, "annotate_deepseekv3"),
            patch.object(
                parallelize,
                "apply_simple_fsdp",
                side_effect=assert_ac_applied_before_simple_fsdp,
            ),
            patch.object(parallelize, "apply_compile", lambda module, **kwargs: module),
            patch.object(parallelize, "maybe_apply_ep_overlap_eager_chunking"),
        ):
            result = parallelize.parallelize_deepseekv3(
                model,
                parallel_dims=FakeParallelDims(),
                training=TrainingConfig(seq_len=1),
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(mode="jit"),
                ac_config=FullAC.Config(),
                dump_folder="dump",
            )

        self.assertIs(result, model)
        self.assertIsNot(model.layers[0], original_layer)
        self.assertTrue(hasattr(model.layers[0], "_checkpoint_wrapped_module"))
        self.assertIs(model.layers[0]._checkpoint_wrapped_module, original_layer)


class TestGraphTrainerLlama3JitActivationCheckpoint(unittest.TestCase):
    def test_llama3_jit_full_ac_wraps_layers(self):
        from torchtitan.config import ParallelismConfig, TrainingConfig
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.llama3 import parallelize

        class FakeParallelDims:
            seq_len_divisor = 1
            cp_enabled = False
            tp_enabled = False

        class TinyLlama3LikeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2)])

        model = TinyLlama3LikeModel()
        original_layer = model.layers[0]

        def assert_ac_applied_before_simple_fsdp(module, **_kwargs):
            self.assertTrue(hasattr(module.layers[0], "_checkpoint_wrapped_module"))
            return module

        with (
            patch.object(parallelize, "annotate_llama"),
            patch.object(
                parallelize,
                "apply_simple_fsdp",
                side_effect=assert_ac_applied_before_simple_fsdp,
            ),
            patch.object(parallelize, "apply_compile", lambda module, **kwargs: module),
        ):
            result = parallelize.parallelize_llama(
                model,
                parallel_dims=FakeParallelDims(),
                training=TrainingConfig(seq_len=1),
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(mode="jit"),
                ac_config=FullAC.Config(),
                dump_folder="dump",
            )

        self.assertIs(result, model)
        self.assertIsNot(model.layers[0], original_layer)
        self.assertTrue(hasattr(model.layers[0], "_checkpoint_wrapped_module"))
        self.assertIs(model.layers[0]._checkpoint_wrapped_module, original_layer)


class _Proj(nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.bias = nn.Parameter(torch.zeros(2)) if bias else None


class TestRegisterParametrizationWrapperNames(unittest.TestCase):
    def tearDown(self):
        for key in [k for k in simple_fsdp._wrap_class_cache if k[0] is _Proj]:
            cls = simple_fsdp._wrap_class_cache.pop(key)
            vars(simple_fsdp).pop(cls.__name__, None)

    @staticmethod
    def _wrap(module):
        names = [n for n, p in module.named_parameters(recurse=False)]
        simple_fsdp._register_parametrization(module, names, nn.Identity())
        return type(module)

    def test_wrappers_of_one_type_get_unique_picklable_names(self):
        unbiased_cls = self._wrap(_Proj(bias=False))
        biased_cls = self._wrap(_Proj(bias=True))

        self.assertNotEqual(unbiased_cls.__name__, biased_cls.__name__)
        for cls in (unbiased_cls, biased_cls):
            self.assertRegex(cls.__name__, r"^SimpleFSDP_Proj_\d+$")
            # pickle saves a class by name; each name must resolve to its own class.
            self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_modules_with_same_params_share_one_wrapper(self):
        self.assertIs(self._wrap(_Proj(bias=True)), self._wrap(_Proj(bias=True)))


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Verify simple_fsdp's MixedPrecisionPolicy actually casts params at NGPU=1."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12358",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_frozen_parameter_remains_frozen(self):
        # SimpleFSDP caches wrapper classes; keep this no-policy fixture isolated.
        class FrozenLinear(nn.Linear):
            pass

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        model = FrozenLinear(8, 8)
        model.weight.requires_grad_(False)
        data_parallel(model, mesh, "fully_shard")
        weight = model._parameters["weight"]
        bias = model._parameters["bias"]
        self.assertFalse(weight.requires_grad)
        self.assertTrue(bias.requires_grad)
        original_weight = weight.to_local().detach().clone()
        original_bias = bias.to_local().detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        model(torch.ones(2, 8)).sum().backward()
        self.assertIsNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        optimizer.step()
        torch.testing.assert_close(weight.to_local(), original_weight, rtol=0, atol=0)
        self.assertFalse(torch.equal(bias.to_local(), original_bias))

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_param_cast_to_bf16_at_ngpu_1(self):
        """With ``mixed_precision_param=bfloat16``, the parametrized weight must
        yield bf16 — and a forward must run in bf16 — even when fsdp /
        dp_replicate / ep are all disabled. Without the unconditional
        simple_fsdp wrap, parameters silently stay in fp32 on a single GPU and
        any downstream bf16-only kernel (e.g. MXFP8) breaks.
        """
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        model = nn.Linear(8, 8)
        self.assertEqual(model.weight.dtype, torch.float32)

        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

        # Parametrization replaces ``weight`` access with a bf16 cast via
        # ``redistribute(forward_dtype=...)``.
        self.assertEqual(model.weight.dtype, torch.bfloat16)

        # Underlying storage stays in fp32 (the cast is applied per forward),
        # confirming this is true mixed precision rather than a one-shot
        # downcast that would lose master-weight precision.
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)

        # End-to-end: forward against the parametrized weight produces bf16
        # activations.
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        y = model(x)
        self.assertEqual(y.dtype, torch.bfloat16)


class TestApplySimpleFSDPExpertDiscovery(unittest.TestCase):
    @patch("torchtitan.experiments.graph_trainer.common_utils.data_parallel")
    def test_full_ac_wrapped_block_shards_experts_on_edp_mesh(self, data_parallel):
        data_parallel.side_effect = lambda module, *_args, **_kwargs: module

        experts = nn.Module()
        experts.num_experts = 64
        moe = nn.Module()
        moe.experts = experts
        block = TransformerBlock()
        block.moe = moe

        model = Decoder.__new__(Decoder)
        nn.Module.__init__(model)
        model.layers = nn.ModuleDict(
            {"0": FullAC.Config().build()._wrap_block(block, base_fqn="layers.0")}
        )

        efsdp_mesh = MagicMock()
        efsdp_mesh.size.return_value = 1
        parallel_dims = MagicMock()
        parallel_dims.dp_replicate_enabled = False
        parallel_dims.dp_shard_enabled = True
        parallel_dims.cp_enabled = False
        parallel_dims.ep_enabled = True
        parallel_dims.ep = 64
        parallel_dims.get_mesh.return_value = MagicMock()
        parallel_dims.get_optional_mesh.return_value = {"efsdp": efsdp_mesh}

        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        result = apply_simple_fsdp(
            model,
            parallel_dims=parallel_dims,
            training=training,
        )

        self.assertIs(result, model)
        self.assertEqual(data_parallel.call_count, 2)
        experts_call, model_call = data_parallel.call_args_list
        self.assertIs(experts_call.args[0], experts)
        self.assertEqual(experts_call.args[1], {"efsdp": efsdp_mesh})
        self.assertEqual(experts_call.args[2], "fully_shard")
        self.assertEqual(experts_call.kwargs["shard_dim"], 0)
        self.assertIs(model_call.args[0], model)
        self.assertIs(model_call.args[1], parallel_dims.get_mesh.return_value)
        self.assertEqual(model_call.args[2], "fully_shard")
        self.assertIs(
            experts_call.kwargs["mp_policy"],
            model_call.kwargs["mp_policy"],
        )


if __name__ == "__main__":
    unittest.main()
