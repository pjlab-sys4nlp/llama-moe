from torch import nn

from .moe_calculators import SwitchDropTokenCalculator, UniversalCalculator
from .moe_experts import LinearExperts, LinearGLUExperts
from .moe_gates import SwitchBalancedGate, TopKBalancedNoisyGate


class LinearMoELayer(nn.Module):
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

        self.gate_type = kwargs["gate_type"]
        self.calculator_type = kwargs["calculator_type"]

        experts = LinearExperts(
            input_size,
            output_size,
            num_experts,
            bias=bias
        )

        if kwargs["gate_type"] == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs["gate_network"],
                use_softmax=kwargs["gate_use_softmax"],
                use_balance=kwargs["gate_use_balance"],
                balance_loss_weight=kwargs["gate_balance_loss_weight"],
                add_noise=kwargs["gate_add_noise"],
                noise_epsilon=kwargs["gate_noise_epsilon"],
            )
        elif kwargs["gate_type"] == "SwitchBalancedGate":  # switch gate
            self.gate = SwitchBalancedGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs["gate_network"],
                use_softmax=kwargs["gate_use_softmax"],
                use_balance=kwargs["gate_use_balance"],
                balance_loss_weight=kwargs["gate_balance_loss_weight"],
            )
        else:
            raise NotImplementedError

        if kwargs["calculator_type"] == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs["multiply_gate_scores"]
            )
        elif kwargs["calculator_type"] == "SwitchDropTokenCalculator":  # switch calculator
            self.calculator = SwitchDropTokenCalculator(
                experts,
                multiply_gate_scores=kwargs["multiply_gate_scores"],
                drop_tokens=kwargs["drop_tokens"],
                capacity_factor=kwargs["capacity_factor"],
            )
        else:
            raise NotImplementedError
        # fmt: on

    def forward(self, x):
        # fmt: off
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)  # shape(batch_size*seq_len, input_size)

        gate_outputs = self.gate(x)  # 计算被选出的专家及其分数，以及gate的loss
        y = self.calculator(x, **gate_outputs)  # 合并各专家的计算结果

        y = y.reshape(original_shape + (self.output_size,))  # shape(batch_size, seq_len, output_size)
        return y, gate_outputs["balance_loss"]
        # fmt: on

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
        if self.gate_type != "TopKBalancedNoisyGate":
            raise ValueError(self.gate_type)
        else:
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
        elif self.input_size != self.output_size:
            raise ValueError(
                'You cannot set "drop_tokens=True" when "input_size != output_size"!'
            )
        else:
            self.calculator.drop_tokens = drop_tokens

    def set_calculator_capacity_factor(self, capacity_factor):
        if self.calculator_type != "SwitchDropTokenCalculator":
            raise ValueError(self.calculator_type)
        else:
            self.calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()


class LinearGLUMoELayer(nn.Module):
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
            **kwargs
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

        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            size_experts=size_experts,
            bias=bias
        )

        if kwargs["gate_type"] == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs["gate_network"],
                use_softmax=kwargs["gate_use_softmax"],
                use_balance=kwargs["gate_use_balance"],
                balance_loss_weight=kwargs["gate_balance_loss_weight"],
                add_noise=kwargs["gate_add_noise"],
                noise_epsilon=kwargs["gate_noise_epsilon"],
            )
        elif kwargs["gate_type"] == "SwitchBalancedGate":  # switch gate
            self.gate = SwitchBalancedGate(
                input_size,
                num_experts,
                num_selects,
                gate_network=kwargs["gate_network"],
                use_softmax=kwargs["gate_use_softmax"],
                use_balance=kwargs["gate_use_balance"],
                balance_loss_weight=kwargs["gate_balance_loss_weight"],
            )
        else:
            raise NotImplementedError

        if kwargs["calculator_type"] == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs["multiply_gate_scores"]
            )
        elif kwargs["calculator_type"] == "SwitchDropTokenCalculator":  # switch calculator
            self.calculator = SwitchDropTokenCalculator(
                experts,
                multiply_gate_scores=kwargs["multiply_gate_scores"],
                drop_tokens=kwargs["drop_tokens"],
                capacity_factor=kwargs["capacity_factor"],
            )
        else:
            raise NotImplementedError
        # fmt: on

    def forward(self, x):
        # fmt: off
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)  # shape(batch_size*seq_len, input_size)

        gate_outputs = self.gate(x)  # 计算被选出的专家及其分数，以及gate的loss
        y = self.calculator(x, **gate_outputs)  # 合并各专家的计算结果

        y = y.reshape(original_shape + (self.output_size,))  # shape(batch_size, seq_len, output_size)
        return y, gate_outputs["balance_loss"]
        # fmt: on

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
        if self.gate_type != "TopKBalancedNoisyGate":
            raise ValueError(self.gate_type)
        else:
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
        elif self.input_size != self.output_size:
            raise ValueError(
                'You cannot set "drop_tokens=True" when "input_size != output_size"!'
            )
        else:
            self.calculator.drop_tokens = drop_tokens

    def set_calculator_capacity_factor(self, capacity_factor):
        if self.calculator_type != "SwitchDropTokenCalculator":
            raise ValueError(self.calculator_type)
        else:
            self.calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()
