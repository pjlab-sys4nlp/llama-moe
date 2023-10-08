""" PyTorch LLaMA-MoE model with residual block (shared experts among all tokens)."""
from torch import nn
from transformers.utils import logging

from smoe.models.llama_moe import LlamaMoEDecoderLayer
from smoe.models.llama_moe.modeling_llama_moe import (
    LlamaMoEForCausalLM,
    LlamaMoEForSequenceClassification,
    LlamaMoEModel,
    LlamaMoEPreTrainedModel,
)
from smoe.models.llama_moe_residual.configuration_llama_moe_residual import (
    LlamaMoEResidualConfig,
)
from smoe.modules.moe_residual.moe_residual_layers import LinearGLUMoEResidualLayer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaMoEResidualConfig"


class LlamaMoEResidualDecoderLayer(LlamaMoEDecoderLayer):
    def __init__(self, config: LlamaMoEResidualConfig, layer_index):
        super(LlamaMoEDecoderLayer, self).__init__(config)
        assert config.intermediate_size == (
                config.intermediate_size_moe + config.intermediate_size_residual
        )

        gating_config = {
            # all gates
            "gate_type": config.gate_type,
            "gate_network": config.gate_network,
            "gate_use_softmax": config.gate_use_softmax,
            "gate_use_balance": config.gate_use_balance,
            "gate_balance_loss_weight": config.gate_balance_loss_weight,
            "gate_add_noise": config.gate_add_noise,
            # TopKBalancedNoisyGate
            "gate_noise_epsilon": config.gate_noise_epsilon,
        }
        calculator_config = {
            # all calculators
            "calculator_type": config.calculator_type,
            "multiply_gate_scores": config.multiply_gate_scores,
            "score_scale_factor": config.score_scale_factor[layer_index] if isinstance(config.score_scale_factor, list) else config.score_scale_factor,
            # SwitchDropTokenCalculator
            "drop_tokens": config.drop_tokens,
            "dropped_padding": config.dropped_padding,
            "capacity_factor": config.capacity_factor,
        }

        self.mlp = LinearGLUMoEResidualLayer(
            input_size=self.hidden_size,
            # ---- different here ---- #
            hidden_size=config.intermediate_size_moe,
            # ------------------------ #
            output_size=self.hidden_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            num_selects=config.num_selects,
            size_experts=(
                config.size_experts[layer_index]
                if config.size_experts is not None
                else None
            ),
            bias=False,
            # ---- different here ---- #
            num_experts_residual=config.num_experts_residual,
            size_experts_residual=config.size_experts_residual[layer_index],
            score_scale_factor_residual=config.score_scale_factor_residual[layer_index] if isinstance(config.score_scale_factor_residual, list) else config.score_scale_factor_residual,
            use_weighting=config.use_weighting,
            # ------------------------ #
            **gating_config,
            **calculator_config,
        )

    def set_moe_residual_calculator_score_scale_factor(self, score_scale_factor):
        self.mlp.set_residual_calculator_score_scale_factor(score_scale_factor)


class LlamaMoEResidualPreTrainedModel(LlamaMoEPreTrainedModel):
    config_class = LlamaMoEResidualConfig
    _no_split_modules = ["LlamaMoEResidualDecoderLayer"]


class LlamaMoEResidualModel(LlamaMoEModel, LlamaMoEResidualPreTrainedModel):
    def __init__(self, config: LlamaMoEResidualConfig):
        super(LlamaMoEModel, self).__init__(config)
        self.layers = nn.ModuleList(
            [
                LlamaMoEResidualDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

    def set_moe_residual_calculator_score_scale_factor(self, score_scale_factor, layer_index=None):
        if layer_index is None:
            for idx, decoder_layer in enumerate(self.layers):
                decoder_layer.set_moe_residual_calculator_score_scale_factor(
                    score_scale_factor
                )
        else:
            self.layers[layer_index].set_moe_residual_calculator_score_scale_factor(
                score_scale_factor
            )


class LlamaMoEResidualForCausalLM(LlamaMoEForCausalLM, LlamaMoEResidualPreTrainedModel):
    def __init__(self, config):
        super(LlamaMoEForCausalLM, self).__init__(config)
        self.model = LlamaMoEResidualModel(config)

    def set_moe_residual_calculator_score_scale_factor(self, score_scale_factor, layer_index=None):
        self.model.set_moe_residual_calculator_score_scale_factor(score_scale_factor, layer_index=layer_index)


class LlamaMoEResidualForSequenceClassification(
    LlamaMoEForSequenceClassification, LlamaMoEResidualPreTrainedModel
):
    def __init__(self, config):
        super(LlamaMoEForSequenceClassification, self).__init__(config)
        self.model = LlamaMoEResidualModel(config)

    def set_moe_residual_calculator_score_scale_factor(self, score_scale_factor, layer_index=None):
        self.model.set_moe_residual_calculator_score_scale_factor(score_scale_factor, layer_index=layer_index)
