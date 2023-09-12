from typing import Optional

from torch import nn

from smoe.modules.moe_residual.residual_blocks import LinearGLU
from smoe.modules.moe.moe_layers import BaseMoELayer, MoEMlpOutput, LinearMoELayer, LinearGLUMoELayer


class BaseMoEResidualLayer(nn.Module):
    def __init__(self):
        super(BaseMoEResidualLayer, self).__init__()

        self.moe_layer: BaseMoELayer
        self.residual_block: nn.Module
        self.weighting_network: Optional[nn.Module] = None

    def forward(self, x) -> MoEMlpOutput:
        moe_output = self.moe_layer(x)
        residual_output = self.residual_block(x)

        if self.weighting_network is not None:
            output_weights = self.weighting_network(x)
            moe_output.hidden_states = moe_output.hidden_states * output_weights[..., 0] + residual_output * output_weights[..., 1]
        else:
            moe_output.hidden_states += residual_output

        return moe_output

    def set_num_selects(self, num_selects):
        self.moe_layer.set_num_selects(num_selects)

    def set_gate_use_softmax(self, use_softmax):
        self.moe_layer.set_gate_use_softmax(use_softmax)

    def set_gate_use_balance(self, use_balance):
        self.moe_layer.set_gate_use_balance(use_balance)

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        self.moe_layer.set_gate_balance_loss_weight(balance_loss_weight)

    def set_gate_add_noise(self, add_noise):
        self.moe_layer.set_gate_add_noise(add_noise)

    def set_gate_noise_epsilon(self, noise_epsilon):
        self.moe_layer.set_gate_noise_epsilon(noise_epsilon)

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.moe_layer.set_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_calculator_drop_tokens(self, drop_tokens):
        self.moe_layer.set_calculator_drop_tokens(drop_tokens)

    def set_calculator_dropped_padding(self, dropped_padding):
        self.moe_layer.set_calculator_dropped_padding(dropped_padding)

    def set_calculator_capacity_factor(self, capacity_factor):
        self.moe_layer.set_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.moe_layer.reset_gate_network()


class LinearMoEResidualLayer(BaseMoEResidualLayer):
    def __init__(
            self, input_size, output_size, num_experts, num_selects, bias=True, use_weighting=True, **kwargs
    ):
        super(LinearMoEResidualLayer, self).__init__()

        self.moe_layer = LinearMoELayer(
            input_size,
            output_size,
            num_experts,
            num_selects,
            bias=bias,
            **kwargs
        )

        self.residual_block = nn.Linear(input_size, output_size, bias=bias)

        if use_weighting:
            self.weighting_network = nn.Sequential(
                nn.Linear(input_size, 2, bias=False),
                nn.Softmax(-1)
            )
        else:
            self.weighting_network = None

    def from_moe_layer(moe_layer, use_weighting=None):
        assert isinstance(moe_layer, LinearMoELayer)
        return LinearMoEResidualLayer(
            moe_layer.input_size,
            moe_layer.output_size,
            moe_layer.num_experts,
            moe_layer.num_selects,
            bias=moe_layer.bias,
            use_weighting=use_weighting,
        )


class LinearGLUMoEResidualLayer(BaseMoEResidualLayer):

    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            num_selects,
            size_experts=None,
            bias=True,
            size_residual=None,
            use_weighting=False,
            moe_layer=None,
            **kwargs,
    ):
        super(LinearGLUMoEResidualLayer, self).__init__()

        if moe_layer is not None:
            self.moe_layer = moe_layer
        else:
            self.moe_layer = LinearGLUMoELayer(
                input_size,
                hidden_size,
                output_size,
                hidden_act,
                num_experts,
                num_selects,
                size_experts=size_experts,
                bias=bias,
                **kwargs
            )

        if size_residual is None:
            size_residual = hidden_size // num_experts
        self.residual_block = LinearGLU(input_size, size_residual, output_size, hidden_act, bias=bias)

        if use_weighting:
            self.weighting_network = nn.Sequential(
                nn.Linear(input_size, 2, bias=False),
                nn.Softmax(-1)
            )
        else:
            self.weighting_network = None

    def from_moe_layer(moe_layer, size_residual=None, use_weighting=None):
        assert isinstance(moe_layer, LinearGLUMoELayer)
        return LinearGLUMoEResidualLayer(
            moe_layer.input_size,
            moe_layer.hidden_size,
            moe_layer.output_size,
            moe_layer.hidden_act,
            moe_layer.num_experts,
            moe_layer.num_selects,
            size_experts=moe_layer.size_experts,
            bias=moe_layer.bias,
            size_residual=size_residual,
            use_weighting=use_weighting,
        )
