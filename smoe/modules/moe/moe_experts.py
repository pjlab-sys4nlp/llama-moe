import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


class LinearExperts(nn.Module):
    """
    Modified from nn.Linear
    """

    __constants__ = ["bias", "in_features", "out_features", "num_experts"]

    def __init__(
        self, in_features, out_features, num_experts, bias=True, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearExperts, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.empty((num_experts, out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((num_experts, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input, i):
        return F.linear(input, self.weight[i], self.bias[i])

    def extra_repr(self):
        return "in_features={}, out_features={}, num_experts={}, bias={}".format(
            self.in_features, self.out_features, self.num_experts, self.bias is not None
        )


class LinearGLUExperts(nn.Module):
    """
    Modified from transformers.models.llama.modeling_llama.LlamaMLP
    """

    __constants__ = [
        "bias",
        "in_features",
        "hidden_features",
        "out_features",
        "hidden_act",
        "num_experts",
        "size_experts",
    ]

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_act,
        num_experts,
        size_experts=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearGLUExperts, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.size_experts = size_experts

        if size_experts is None:  # all experts share the same number of hidden neurons
            assert hidden_features % num_experts == 0
            size_per_expert = hidden_features // num_experts
            size_experts = [size_per_expert for _ in range(num_experts)]
        else:  # use specified expert sizes
            assert (
                len(size_experts) == num_experts
                and sum(size_experts) == hidden_features
            )

        self.act_fn = ACT2FN[hidden_act]

        self.weight_gate = nn.ParameterList()
        self.weight_up = nn.ParameterList()
        self.weight_down = nn.ParameterList()

        for i in range(num_experts):
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_gate = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_up = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_down = nn.Parameter(
                torch.empty((out_features, size_experts[i]), **factory_kwargs)
            )
            self.weight_gate.append(this_expert_weight_gate)
            self.weight_up.append(this_expert_weight_up)
            self.weight_down.append(this_expert_weight_down)

        if bias:
            self.bias_gate = nn.ParameterList()
            self.bias_up = nn.ParameterList()
            self.bias_down = nn.ParameterList()

            for i in range(num_experts):
                this_expert_bias_gate = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_up = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_down = nn.Parameter(
                    torch.empty((out_features,), **factory_kwargs)
                )
                self.bias_gate.append(this_expert_bias_gate)
                self.bias_up.append(this_expert_bias_up)
                self.bias_down.append(this_expert_bias_down)
        else:
            self.register_parameter("bias_gate", None)
            self.register_parameter("bias_up", None)
            self.register_parameter("bias_down", None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight_gate[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_up[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_down[i], a=math.sqrt(5))
            if self.bias_gate is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_gate[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_gate[i], -bound, bound)
            if self.bias_up is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_up[i], -bound, bound)
            if self.bias_down is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_down[i], -bound, bound)

    def forward(self, input, i):
        gate = self.act_fn(
            F.linear(
                input,
                self.weight_gate[i],
                self.bias_gate[i] if self.bias_gate is not None else None,
            )
        )
        up = F.linear(
            input,
            self.weight_up[i],
            self.bias_up[i] if self.bias_up is not None else None,
        )
        down = F.linear(
            gate * up,
            self.weight_down[i],
            self.bias_down[i] if self.bias_down is not None else None,
        )
        return down

    def extra_repr(self):
        return (
            "in_features={}, hidden_features={}, out_features={}, hidden_act={},"
            " num_experts={}, size_experts={}, bias={}".format(
                self.in_features,
                self.hidden_features,
                self.out_features,
                self.hidden_act,
                self.num_experts,
                self.size_experts,
                self.bias_gate is not None,
            )
        )
