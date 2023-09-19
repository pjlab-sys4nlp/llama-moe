import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
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
            this_expert_weight_gate = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )  # this matrix will be transposed when performing linear forwarding
            this_expert_weight_up = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )  # this matrix will be transposed when performing linear forwarding
            this_expert_weight_down = nn.Parameter(
                torch.empty((out_features, size_experts[i]), **factory_kwargs)
            )  # this matrix will be transposed when performing linear forwarding
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


class SoftGLUExperts(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hidden_act,
        num_experts: int,
        bias: bool = True,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.bias = bias

        self.act_fn = ACT2FN[hidden_act]

        self.weight_gate = nn.Parameter(
            torch.empty(
                (num_experts, in_features, hidden_features), device=device, dtype=dtype
            )
        )
        self.weight_up = nn.Parameter(
            torch.empty(
                (num_experts, in_features, hidden_features), device=device, dtype=dtype
            )
        )
        self.weight_down = nn.Parameter(
            torch.empty(
                (num_experts, hidden_features, out_features), device=device, dtype=dtype
            )
        )

        bias_gate_param: Optional[nn.Parameter] = None
        bias_up_param: Optional[nn.Parameter] = None
        bias_down_param: Optional[nn.Parameter] = None

        if self.bias:
            bias_gate_param = nn.Parameter(
                torch.empty((num_experts, hidden_features), device=device, dtype=dtype)
            )
            bias_up_param = nn.Parameter(
                torch.empty((num_experts, hidden_features), device=device, dtype=dtype)
            )
            bias_down_param = nn.Parameter(
                torch.empty((num_experts, out_features), device=device, dtype=dtype)
            )

        self.bias_gate: Optional[nn.Parameter]
        self.bias_up: Optional[nn.Parameter]
        self.bias_down: Optional[nn.Parameter]

        self.register_parameter("bias_gate", bias_gate_param)
        self.register_parameter("bias_up", bias_up_param)
        self.register_parameter("bias_down", bias_down_param)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Mostly copy-pasta from 'nn.Linear.reset_parameters'
        #
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight_gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_up, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_down, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_gate)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_gate, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_up, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_down, -bound, bound)

    def forward(self, x):
        print(1000000000, x.size())
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input with embed_dim={self.in_features} (dim=-1), but "
                f"found {x.size(-1)}"
            )
        elif x.size(1) != self.num_experts:
            raise ValueError(
                f"Expected input with num_experts={self.num_experts} (dim=1), but "
                f"found {x.size(1)}"
            )

        # NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
        # work correctly, we have to give them different names.
        # x = einsum(x, self.weight, "b n ... d1, n d1 d2 -> b n ... d2")
        gate = self.act_fn(
            einsum(x, self.weight_gate, "b n ... di, n di dh -> b n ... dh")
        )
        print(11111111, gate.size())
        if self.bias_gate is not None:
            if gate.ndim == 3:
                bias_gate = rearrange(self.bias_gate, "n d -> () n d")
            elif gate.ndim == 4:
                bias_gate = rearrange(self.bias_gate, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {gate.ndim}"
                )
            gate = gate + bias_gate

        up = einsum(x, self.weight_up, "b n ... di, n di dh -> b n ... dh")
        print(22222222, up.size())
        if self.bias_up is not None:
            if up.ndim == 3:
                bias_up = rearrange(self.bias_up, "n d -> () n d")
            elif up.ndim == 4:
                bias_up = rearrange(self.bias_up, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {up.ndim}"
                )
            up = up + bias_up

        down = einsum(gate * up, self.weight_down, "b n ... di, n di do -> b n ... do")
        print(333333333, down.size())
        if self.bias_down is not None:
            if down.ndim == 3:
                bias_down = rearrange(self.bias_down, "n d -> () n d")
            elif down.ndim == 4:
                bias_down = rearrange(self.bias_down, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {down.ndim}"
                )
            down = down + bias_down

        return down

    def extra_repr(self) -> str:
        return (
            "in_features={}, hidden_features={}, out_features={}, hidden_act={},"
            " num_experts={}, bias={}".format(
                self.in_features,
                self.hidden_features,
                self.out_features,
                self.hidden_act,
                self.num_experts,
                self.bias_gate is not None,
            )
        )
