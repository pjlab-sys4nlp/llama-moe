""" LLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LlamaMoEResidualConfig(PretrainedConfig):
    model_type = "llama_moe_residual"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,  # 688*16
        # ---- different here ---- #
        intermediate_size_moe=9632,  # 688*14
        intermediate_size_residual=1376,  # 688*2
        # ------------------------ #
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        #### -------- moe expert configs -------- ####
        num_experts=14,
        num_selects=2,
        size_experts=None,
        #### -------- moe residual block configs -------- ####
        # ---- different here ---- #
        num_experts_residual=2,
        size_experts_residual=None,
        score_scale_factor_residual=1.0,
        use_weighting=False,
        # ------------------------ #
        #### -------- moe gate configs -------- ####
        gate_type="TopKBalancedNoisyGate",
        gate_network="mlp",
        gate_use_softmax=True,
        gate_use_balance=True,
        gate_balance_loss_weight=1e-2,
        gate_add_noise=True,
        # TopKBalancedNoisyGate
        gate_noise_epsilon=1e-2,
        #### -------- moe calculator configs -------- ####
        calculator_type="UniversalCalculator",
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        # SwitchDropTokenCalculator
        drop_tokens=True,
        dropped_padding="zero",
        capacity_factor=1.25,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # ---- different here ---- #
        self.intermediate_size_moe = intermediate_size_moe
        self.intermediate_size_residual = intermediate_size_residual
        # ------------------------ #
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts

        # ---- different here ---- #
        self.num_experts_residual = num_experts_residual
        self.size_experts_residual = size_experts_residual
        self.score_scale_factor_residual = score_scale_factor_residual
        self.use_weighting = use_weighting
        # ------------------------ #

        self.gate_type = gate_type
        self.gate_network = gate_network
        self.gate_use_softmax = gate_use_softmax
        self.gate_use_balance = gate_use_balance
        self.gate_balance_loss_weight = gate_balance_loss_weight
        self.gate_add_noise = gate_add_noise
        self.gate_noise_epsilon = gate_noise_epsilon

        self.calculator_type = calculator_type
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.drop_tokens = drop_tokens
        self.dropped_padding = dropped_padding
        self.capacity_factor = capacity_factor

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}"
            )
