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
            "score_scale_factor": (
                config.score_scale_factor[layer_index]
                if isinstance(config.score_scale_factor, list)
                else config.score_scale_factor
            ),
            # SwitchDropTokenCalculator
            "drop_tokens": config.drop_tokens,
            "dropped_padding": config.dropped_padding,
            "capacity_factor": config.capacity_factor,
        }
        residual_config = {
            "gate_use_softmax_residual": config.gate_use_softmax_residual,
            "multiply_gate_scores_residual": config.multiply_gate_scores_residual,
            "score_scale_factor_residual": (
                config.score_scale_factor_residual[layer_index]
                if isinstance(config.score_scale_factor_residual, list)
                else config.score_scale_factor_residual
            ),
        }

        self.mlp = LinearGLUMoEResidualLayer(
            input_size=self.hidden_size,
            # ↓↓↓↓ different here ↓↓↓↓ #
            hidden_size=config.intermediate_size_moe,
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #
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
            # ↓↓↓↓ different here ↓↓↓↓ #
            num_experts_residual=config.num_experts_residual,
            size_experts_residual=(
                config.size_experts_residual[layer_index]
                if config.size_experts is not None
                else None
            ),
            use_weighting=config.use_weighting,
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #
            **gating_config,
            **calculator_config,
            # ↓↓↓↓ different here ↓↓↓↓ #
            **residual_config,
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #
        )

    def set_moe_residual_gate_use_softmax(self, use_softmax):
        self.mlp.set_residual_gate_use_softmax(use_softmax)

    def set_moe_residual_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.mlp.set_residual_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_residual_calculator_score_scale_factor(self, score_scale_factor):
        self.mlp.set_residual_calculator_score_scale_factor(score_scale_factor)

    def reset_residual_experts(self):
        self.mlp.reset_residual_experts()


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

    def update_config(self):
        self.config.vocab_size = self.config.vocab_size
        self.config.max_position_embeddings = self.config.max_position_embeddings
        # ↓↓↓↓↓↓↓↓↓↓↓↓ changed here ↓↓↓↓↓↓↓↓↓↓↓↓ #
        self.config.hidden_size = self.layers[0].mlp.moe_layer.input_size
        self.config.intermediate_size_moe = self.layers[0].mlp.moe_layer.hidden_size
        self.config.intermediate_size_residual = self.layers[
            0
        ].mlp.residual_block.hidden_size
        self.config.intermediate_size = (
            self.config.intermediate_size_moe + self.config.intermediate_size_residual
        )
        self.config.num_hidden_layers = len(self.layers)
        self.config.num_attention_heads = self.layers[0].self_attn.num_heads
        self.config.hidden_act = self.layers[0].mlp.moe_layer.hidden_act
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #
        self.config.initializer_range = self.config.initializer_range
        self.config.rms_norm_eps = self.config.rms_norm_eps
        self.config.pretraining_tp = self.config.pretraining_tp
        self.config.use_cache = self.config.use_cache
        self.config.rope_scaling = self.config.rope_scaling
        self.config._rope_scaling_validation()

        self.config.num_experts = self.layers[0].mlp.moe_layer.num_experts
        self.config.num_selects = self.layers[0].mlp.moe_layer.num_selects
        self.config.size_experts = [
            self.layers[i].mlp.moe_layer.calculator.experts.size_experts
            for i in range(self.config.num_hidden_layers)
        ]

        self.config.gate_type = vars(self.layers[0].mlp.moe_layer).get(
            "gate_type", "TopKBalancedNoisyGate"
        )
        self.config.gate_network = vars(self.layers[0].mlp.moe_layer.gate).get(
            "gate_network_type", "mlp"
        )
        self.config.gate_use_softmax = vars(self.layers[0].mlp.moe_layer.gate).get(
            "use_softmax", True
        )
        self.config.gate_use_balance = vars(self.layers[0].mlp.moe_layer.gate).get(
            "use_balance", True
        )
        self.config.gate_balance_loss_weight = vars(
            self.layers[0].mlp.moe_layer.gate
        ).get("balance_loss_weight", 1e-2)
        self.config.gate_add_noise = vars(self.layers[0].mlp.moe_layer.gate).get(
            "add_noise", True
        )
        self.config.gate_noise_epsilon = vars(self.layers[0].mlp.moe_layer.gate).get(
            "noise_epsilon", 1e-2
        )

        self.config.calculator_type = vars(self.layers[0].mlp.moe_layer).get(
            "calculator_type", "UniversalCalculator"
        )
        self.config.multiply_gate_scores = vars(
            self.layers[0].mlp.moe_layer.calculator
        ).get("multiply_gate_scores", True)
        self.config.score_scale_factor = [
            vars(self.layers[i].mlp.moe_layer.calculator).get("score_scale_factor", 1.0)
            for i in range(self.config.num_hidden_layers)
        ]
        self.config.drop_tokens = vars(self.layers[0].mlp.moe_layer.calculator).get(
            "drop_tokens", True
        )
        self.config.dropped_padding = vars(self.layers[0].mlp.moe_layer.calculator).get(
            "dropped_padding", "zero"
        )
        self.config.capacity_factor = vars(self.layers[0].mlp.moe_layer.calculator).get(
            "capacity_factor", 1.25
        )

        # ↓↓↓↓ different here ↓↓↓↓ #
        self.config.num_experts_residual = self.layers[0].mlp.residual_block.num_experts
        self.config.size_experts_residual = [
            self.layers[i].mlp.residual_block.calculator.experts.size_experts
            for i in range(self.config.num_hidden_layers)
        ]
        self.config.gate_use_softmax_residual = vars(
            self.layers[0].mlp.residual_block.gate
        ).get("use_softmax", True)
        self.config.multiply_gate_scores_residual = vars(
            self.layers[0].mlp.residual_block.calculator
        ).get("multiply_gate_scores", True)
        self.config.score_scale_factor_residual = [
            vars(self.layers[i].mlp.residual_block.calculator).get(
                "score_scale_factor", 1.0
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.config.use_weighting = "weighting_network" in vars(
            self.layers[0].mlp.residual_block.gate
        )
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #

    def set_moe_residual_gate_use_softmax(self, use_softmax):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_residual_gate_use_softmax(use_softmax)

    def set_moe_residual_calculator_multiply_gate_scores(self, multiply_gate_scores):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_residual_calculator_multiply_gate_scores(
                multiply_gate_scores
            )

    def set_moe_residual_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        if layer_index is None:
            for idx, decoder_layer in enumerate(self.layers):
                decoder_layer.set_moe_residual_calculator_score_scale_factor(
                    score_scale_factor
                )
        else:
            self.layers[layer_index].set_moe_residual_calculator_score_scale_factor(
                score_scale_factor
            )

    def reset_residual_experts(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_residual_experts()


class LlamaMoEResidualForCausalLM(LlamaMoEForCausalLM, LlamaMoEResidualPreTrainedModel):
    def __init__(self, config):
        super(LlamaMoEForCausalLM, self).__init__(config)
        self.model = LlamaMoEResidualModel(config)

    def update_config(self):
        self.model.update_config()

    def set_moe_residual_gate_use_softmax(self, use_softmax):
        self.model.set_moe_residual_gate_use_softmax(use_softmax)

    def set_moe_residual_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.model.set_moe_residual_calculator_multiply_gate_scores(
            multiply_gate_scores
        )

    def set_moe_residual_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        self.model.set_moe_residual_calculator_score_scale_factor(
            score_scale_factor, layer_index=layer_index
        )

    def reset_residual_experts(self):
        self.model.reset_residual_experts()


class LlamaMoEResidualForSequenceClassification(
    LlamaMoEForSequenceClassification, LlamaMoEResidualPreTrainedModel
):
    def __init__(self, config):
        super(LlamaMoEForSequenceClassification, self).__init__(config)
        self.model = LlamaMoEResidualModel(config)

    def update_config(self):
        self.model.update_config()

    def set_moe_residual_gate_use_softmax(self, use_softmax):
        self.model.set_moe_residual_gate_use_softmax(use_softmax)

    def set_moe_residual_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.model.set_moe_residual_calculator_multiply_gate_scores(
            multiply_gate_scores
        )

    def set_moe_residual_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        self.model.set_moe_residual_calculator_score_scale_factor(
            score_scale_factor, layer_index=layer_index
        )

    def reset_residual_experts(self):
        self.model.reset_residual_experts()
