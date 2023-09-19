import torch

from smoe.modules.moe.moe_layers import LinearGLUMoELayer

input_size = 128
hidden_size = 4096
output_size = 128
hidden_act = "silu"
num_experts = 16
num_selects = 1
size_experts = None
bias = True

gating_config = {
    "gate_type": "SoftMoEGate",
    "slots_per_expert": 1,
}

calculator_config = {
    "calculator_type": "SoftMoECalculator",
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

batch_size = 8
toekn_len = 96

input = torch.rand((batch_size, toekn_len, input_size))
output = layer(input)
