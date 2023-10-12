import warnings
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
from .moe_gates import (
    RandomLearnableGate,
    SwitchBalancedGate,
    TopKBalancedNoisyGate,
    UniformPlainGate,
)


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

    def _create_gate(self, **kwargs):
        self.gate_type = kwargs.get("gate_type", "TopKBalancedNoisyGate")

        if self.gate_type == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        elif self.gate_type == "SwitchBalancedGate":  # switch gate
            self.gate = SwitchBalancedGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
            )
        elif self.gate_type == "UniformPlainGate":  # all select gate
            self.gate = UniformPlainGate(
                self.input_size,
                self.num_experts,
                use_softmax=kwargs.get("gate_use_softmax", False),
            )
        elif self.gate_type == "RandomLearnableGate":  # random gate with network
            self.gate = RandomLearnableGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", False),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        else:
            raise NotImplementedError

    def _create_calculator(self, experts, **kwargs):
        self.calculator_type = kwargs.get("calculator_type", "UniversalCalculator")

        if self.calculator_type == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
            )
        elif self.calculator_type == "SwitchDropTokenCalculator":  # switch calculator
            self.calculator = SwitchDropTokenCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
                drop_tokens=kwargs.get("drop_tokens", True),
                dropped_padding=kwargs.get("dropped_padding", "zero"),
                capacity_factor=kwargs.get("capacity_factor", 1.25),
            )
        else:
            raise NotImplementedError

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

    # fmt: off
    def set_num_selects(self, num_selects):
        if "num_selects" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "num_selects".')
        elif num_selects > self.gate.num_experts:
            raise ValueError('The value of "num_selects" must satisfy "num_selects <= num_experts"!')
        elif self.gate_type in ("SwitchBalancedGate",):
            raise ValueError(f"{self.gate_type} doesn't support manually setting num_selects.")
        else:
            self.num_selects = num_selects
            self.gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        if "use_softmax" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "use_softmax".')
        else:
            self.gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        if "use_balance" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "use_balance".')
        else:
            self.gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        if "balance_loss_weight" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "balance_loss_weight".')
        else:
            self.gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        if "add_noise" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "add_noise".')
        else:
            self.gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        if "noise_epsilon" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "noise_epsilon".')
        else:
            self.gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        if "multiply_gate_scores" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "multiply_gate_scores".')
        else:
            self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        if "score_scale_factor" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "score_scale_factor".')
        else:
            self.calculator.score_scale_factor = score_scale_factor

    def set_calculator_drop_tokens(self, drop_tokens):
        if "drop_tokens" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "drop_tokens".')
        elif drop_tokens and self.calculator.dropped_padding != "zero" and self.input_size != self.output_size:
            warnings.warn('Setting "drop_tokens=True" without zero dropped padding when "input_size != output_size" will cause error!')
        else:
            self.calculator.drop_tokens = drop_tokens

    def set_calculator_dropped_padding(self, dropped_padding):
        if "dropped_padding" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "dropped_padding".')
        elif dropped_padding not in self.calculator.available_dropped_padding_choices:
            raise ValueError(f"'dropped_padding' type not available! (available choices: {self.calculator.available_dropped_padding_choices})")
        elif self.calculator.drop_tokens and dropped_padding != "zero" and self.input_size != self.output_size:
            warnings.warn(f'Setting "dropped_padding={dropped_padding}" with "drop_tokens=True" when "input_size != output_size" will cause error!')
        else:
            self.calculator.dropped_padding = dropped_padding

    def set_calculator_capacity_factor(self, capacity_factor):
        if "capacity_factor" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "capacity_factor".')
        else:
            self.calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()
    # fmt: on


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
        self.bias = bias

        experts = LinearExperts(
            input_size,
            output_size,
            num_experts,
            bias=bias,
        )

        self._create_gate(**kwargs)
        self._create_calculator(experts, **kwargs)
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
        self.size_experts = size_experts
        self.bias = bias

        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            size_experts=size_experts,
            bias=bias
        )

        self._create_gate(**kwargs)
        self._create_calculator(experts, **kwargs)
        # fmt: on
