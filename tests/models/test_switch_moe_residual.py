import torch

from smoe.modules.moe_residual.moe_residual_layers import LinearGLUMoEResidualLayer


def test_switch_moe_residual():
    input_size = 4096
    hidden_size = 688 * 13
    output_size = 4096
    hidden_act = "silu"
    num_experts = 13
    num_selects = 1
    size_experts = None
    bias = True

    num_experts_residual = 3
    size_experts_residual = None  # 688
    score_scale_factor_residual = 12.0
    use_weighting = False

    gating_config = {
        "gate_type": "SwitchBalancedGate",
        "gate_network": "mlp",
        "gate_use_softmax": True,
        "gate_use_balance": True,
        "gate_balance_loss_weight": 0.01,
        "gate_add_noise": False,
    }

    calculator_config = {
        "calculator_type": "SwitchDropTokenCalculator",
        "multiply_gate_scores": True,
        "score_scale_factor": 4.0,
        "drop_tokens": True,
        "capacity_factor": 1.25,
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
    test_switch_moe_residual()
