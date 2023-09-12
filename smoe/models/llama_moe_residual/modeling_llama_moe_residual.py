""" PyTorch LLaMA-MoE model with residual block (shared param among experts)."""
from torch import nn
from transformers.utils import logging

from smoe.models.llama_moe import LlamaMoEDecoderLayer
from smoe.models.llama_moe.modeling_llama_moe import  LlamaMoEPreTrainedModel, LlamaMoEModel, LlamaMoEForCausalLM, LlamaMoEForSequenceClassification
from smoe.models.llama_moe_residual.configuration_llama_moe_residual import LlamaMoEResidualConfig
from smoe.modules.moe_residual.moe_residual_layers import LinearGLUMoEResidualLayer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaMoEResidualConfig"


class LlamaMoEResidualDecoderLayer(LlamaMoEDecoderLayer):
    def __init__(self, config: LlamaMoEResidualConfig, layer_index):
        super().__init__(config, layer_index)

        self.mlp = LinearGLUMoEResidualLayer.from_moe_layer(
            self.mlp,
            size_residual=config.size_residual,
            use_weighting=config.residual_use_weighting,
        )


class LlamaMoEResidualPreTrainedModel(LlamaMoEPreTrainedModel):
    config_class = LlamaMoEResidualConfig
    _no_split_modules = ["LlamaMoEResidualDecoderLayer"]


class LlamaMoEResidualModel(LlamaMoEModel, LlamaMoEResidualPreTrainedModel):
    def __init__(self, config: LlamaMoEResidualConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaMoEResidualDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


class LlamaMoEResidualForCausalLM(LlamaMoEForCausalLM, LlamaMoEResidualPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMoEResidualModel(config)


class LlamaMoEResidualForSequenceClassification(
    LlamaMoEForSequenceClassification, LlamaMoEResidualPreTrainedModel
):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMoEResidualModel(config)
