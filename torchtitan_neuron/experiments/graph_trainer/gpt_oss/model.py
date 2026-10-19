# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from dataclasses import dataclass

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)
from torchtitan.models.gpt_oss import GptOssModel


class GraphTrainerGptOssModel(GptOssModel):
    @dataclass(kw_only=True, slots=True)
    class Config(GptOssModel.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_weights(self, *args, **kwargs):
        with disable_active_parametrization():
            super().init_weights(*args, **kwargs)
