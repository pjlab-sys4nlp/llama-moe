import torch

from smoe.modules.moe_residual.moe_residual_layers import LinearGLUMoEResidualLayer

def test_noise_moe_residual():
    input_size = 4096
    hidden_size = 688 * 14
    output_size = 4096
    hidden_act = "silu"
    num_experts = 14
    num_selects = 2
    size_experts = None
    bias = True

    num_experts_residual = 2
    size_experts_residual = None  # 688
    score_scale_factor_residual = 8.0
    use_weighting = False

    gating_config = {
        "gate_type": "TopKBalancedNoisyGate",
        "gate_network": "mlp",
        "gate_use_softmax": True,
        "gate_use_balance": True,
        "gate_balance_loss_weight": 0.01,
        "gate_add_noise": True,
        "gate_noise_epsilon": 0.01,
    }

    calculator_config = {
        "calculator_type": "UniversalCalculator",
        "multiply_gate_scores": True,
        "score_scale_factor": 8.0,
    }

    layer = LinearGLUMoEResidualLayer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        hidden_act=hidden_act,
        num_experts=num_experts,
        num_selects=num_selects,
        size_experts=size_experts,
        bias=bias,
        num_experts_residual=num_experts_residual,
        size_experts_residual=size_experts_residual,
        score_scale_factor_residual=score_scale_factor_residual,
        use_weighting=use_weighting,
        **gating_config,
        **calculator_config,
    )

    batch_size = 64

    input = torch.rand((batch_size, input_size))
    output = layer(input)

if __name__ == "__main__":
    test_noise_moe_residual()