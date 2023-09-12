import torch

from smoe.modules.moe_residual.moe_residual_layers import LinearGLUMoEResidualLayer

input_size = 128
hidden_size = 4096
output_size = 128
hidden_act = "silu"
num_experts = 16
num_selects = 1
size_experts = None
bias = True

size_residual = None
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
    size_residual=size_residual,
    use_weighting=use_weighting,
    **gating_config,
    **calculator_config,
)

batch_size = 64

input = torch.rand((batch_size, input_size))
output = layer(input)
