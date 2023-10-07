from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from transformers.utils import ModelOutput

from .moe_calculators import (
    CalculatorOutput,
    SwitchDropTokenCalculator,
    UniversalCalculator,
)
from .moe_experts import LinearExperts, LinearGLUExperts
from .moe_gates import SwitchBalancedGate, TopKBalancedNoisyGate


@dataclass
class MoEMlpOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    gate_load: Optional[list] = None
    gate_importance: Optional[list] = None


class BaseMoELayer(nn.Module):
    def __init__(self):
        super(BaseMoELayer, self).__init__()

        self.gate: Union[SwitchBalancedGate, TopKBalancedNoisyGate]
        self.calculator: Union[SwitchDropTokenCalculator, UniversalCalculator]

    def forward(self, x) -> MoEMlpOutput:
        original_shape = x.shape[:-1]
        # shape(batch_size*seq_len, input_size)
        x = x.reshape(-1, self.input_size)

        # 计算被选出的专家及其分数，以及gate的loss
        gate_outputs: dict = self.gate(x)
        # 合并各专家的计算结果
        calc_outs: CalculatorOutput = self.calculator(x, **gate_outputs)
        y = calc_outs.hidden_states
        # shape(batch_size, seq_len, output_size)
        y = y.reshape(original_shape + (self.output_size,))

        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=gate_outputs.get("balance_loss"),
            num_dropped_tokens=calc_outs.num_dropped_tokens,
            gate_load=gate_outputs.get("load", torch.tensor(-1)),
            gate_importance=gate_outputs.get("importance", torch.tensor(-1)),
        )

    def set_num_selects(self, num_selects):
        if num_selects > self.gate.num_experts:
            raise ValueError(
                'The value of "num_selects" must satisfy "num_selects <= num_experts"!'
            )
        elif self.gate_type == "SwitchBalancedGate":
            raise ValueError(
                "SwitchBalancedGate doesn't support manually setting num_selects."
            )
        else:
            self.gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        self.gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        self.gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        self.gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        self.gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        if self.gate_type != "TopKBalancedNoisyGate":
            raise ValueError(self.gate_type)
        else:
            self.gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_drop_tokens(self, drop_tokens):
        if self.calculator_type != "SwitchDropTokenCalculator":
            raise ValueError(self.calculator_type)
        elif (
            drop_tokens
            and self.calculator.dropped_padding != "zero"
            and self.input_size != self.output_size
        ):
            raise Warning(
                'Setting "drop_tokens=True" without zero dropped padding when "input_size != output_size" will cause error!'
            )
        else:
            self.calculator.drop_tokens = drop_tokens

    def set_calculator_dropped_padding(self, dropped_padding):
        if self.calculator_type != "SwitchDropTokenCalculator":
            raise ValueError(self.calculator_type)
        elif dropped_padding not in self.calculator.available_dropped_padding_choices:
            raise ValueError(
                f"'dropped_padding' type not available! (available choices: {self.calculator.available_dropped_padding_choices})"
            )
        elif (
            self.calculator.drop_tokens
            and dropped_padding != "zero"
            and self.input_size != self.output_size
        ):
            raise Warning(
                'Setting "drop_tokens=True" without zero dropped padding when "input_size != output_size" will cause error!'
            )
        else:
            self.calculator.dropped_padding = dropped_padding

    def set_calculator_capacity_factor(self, capacity_factor):
        if self.calculator_type != "SwitchDropTokenCalculator":
            raise ValueError(self.calculator_type)
        else:
            self.calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()


class LinearMoELayer(BaseMoELayer):
    def __init__(
        self, input_size, output_size, num_experts, num_selects, bias=True, **kwargs
    ):
        # fmt: off
        super(LinearMoELayer, self).__init__()
        assert (num_selects <= num_experts)  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_type = kwargs.get("gate_type", "TopKBalancedNoisyGate")
        self.calculator_type = kwargs.get("calculator_type", "UniversalCalculator")

        experts = LinearExperts(
            input_size,
            output_size,
            num_experts,
            bias=bias
        )

        if self.gate_type == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        elif self.gate_type == "SwitchBalancedGate":  # switch gate
            self.gate = SwitchBalancedGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
            )
        else:
            raise NotImplementedError

        if self.calculator_type == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            )
        elif self.calculator_type == "SwitchDropTokenCalculator":  # switch calculator
            self.calculator = SwitchDropTokenCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                drop_tokens=kwargs.get("drop_tokens", True),
                dropped_padding=kwargs.get("dropped_padding", "zero"),
                capacity_factor=kwargs.get("capacity_factor", 1.25),
            )
        else:
            raise NotImplementedError
        # fmt: on


class LinearGLUMoELayer(BaseMoELayer):
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
        **kwargs,
    ):
        # fmt: off
        super(LinearGLUMoELayer, self).__init__()
        assert (num_selects <= num_experts)  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_type = kwargs.get("gate_type", "TopKBalancedNoisyGate")
        self.calculator_type = kwargs.get("calculator_type", "UniversalCalculator")

        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            size_experts=size_experts,
            bias=bias
        )

        if self.gate_type == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        elif self.gate_type == "SwitchBalancedGate":  # switch gate
            self.gate = SwitchBalancedGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
            )
        else:
            raise NotImplementedError

        if self.calculator_type == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            )
        elif self.calculator_type == "SwitchDropTokenCalculator":  # switch calculator
            self.calculator = SwitchDropTokenCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                drop_tokens=kwargs.get("drop_tokens", True),
                dropped_padding=kwargs.get("dropped_padding", "zero"),
                capacity_factor=kwargs.get("capacity_factor", 1.25),
            )
        else:
            raise NotImplementedError
        # fmt: on
