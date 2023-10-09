import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


class LinearGLU(nn.Module):
    """
    Modified from transformers.models.llama.modeling_llama.LlamaMLP
    """

    __constants__ = [
        "bias",
        "in_features",
        "hidden_features",
        "out_features",
        "hidden_act",
    ]

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_act,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearGLU, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act

        self.act_fn = ACT2FN[hidden_act]

        self.weight_gate = nn.Parameter(
            torch.empty((hidden_features, in_features), **factory_kwargs)
        )
        self.weight_up = nn.Parameter(
            torch.empty((hidden_features, in_features), **factory_kwargs)
        )
        self.weight_down = nn.Parameter(
            torch.empty((out_features, hidden_features), **factory_kwargs)
        )

        if bias:
            self.bias_gate = nn.Parameter(
                torch.empty((hidden_features,), **factory_kwargs)
            )
            self.bias_up = nn.Parameter(
                torch.empty((hidden_features,), **factory_kwargs)
            )
            self.bias_down = nn.Parameter(
                torch.empty((out_features,), **factory_kwargs)
            )
        else:
            self.register_parameter("bias_gate", None)
            self.register_parameter("bias_up", None)
            self.register_parameter("bias_down", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_up, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_down, a=math.sqrt(5))
        if self.bias_gate is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_gate)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_gate, -bound, bound)
        if self.bias_up is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_up, -bound, bound)
        if self.bias_down is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_down, -bound, bound)

    def forward(self, input):
        gate = self.act_fn(F.linear(input, self.weight_gate, self.bias_gate))
        up = F.linear(input, self.weight_up, self.bias_up)
        down = F.linear(gate * up, self.weight_down, self.bias_down)
        return down

    def extra_repr(self):
        return "in_features={}, hidden_features={}, out_features={}, hidden_act={}, bias={}".format(
            self.in_features,
            self.hidden_features,
            self.out_features,
            self.hidden_act,
            self.bias_gate is not None,
        )
