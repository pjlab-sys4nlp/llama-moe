import types

import torch

from smoe.modules.moe.moe_layers import LinearGLUMoELayer
from smoe.utils.model_operation.change_llama_moe_forward import (
    forward_topk_balanced_noisy_gate_with_random_expert_selection,
)
from smoe.utils.seed import set_seed


def test_noise_moe():
    input_size = 128
    hidden_size = 4096
    output_size = 128
    hidden_act = "silu"
    num_experts = 16
    num_selects = 16
    size_experts = None
    bias = True

    gating_config = {
        "gate_type": "TopKBalancedNoisyGate",
        "gate_network": "mlp",
        "gate_use_softmax": True,
        "gate_use_balance": True,
        "gate_balance_loss_weight": 0.01,
        "gate_add_noise": True,
        "gate_noise_epsilon": 1e-2,
    }

    calculator_config = {
        "calculator_type": "UniversalCalculator",
        "multiply_gate_scores": False,
        "score_scale_factor": 1.0,
    }

    layer = LinearGLUMoELayer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        hidden_act=hidden_act,
        num_experts=num_experts,
        num_selects=num_selects,
        size_experts=size_experts,
        bias=bias,
        **gating_config,
        **calculator_config,
    )

    batch_size = 64

    layer.gate.forward = types.MethodType(
        forward_topk_balanced_noisy_gate_with_random_expert_selection, layer.gate
    )
    set_seed(0)

    input = torch.rand((batch_size, input_size))
    output = layer(input)


if __name__ == "__main__":
    test_noise_moe()
