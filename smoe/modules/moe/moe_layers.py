from torch import nn

from .moe_calculators import UniversalCalculator
from .moe_experts import LinearExperts, LinearGLUExperts
from .moe_gates import TopKBalancedNoisyGate


class LinearMoELayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        num_selects,
        bias=True,
        gate_network="mlp",
        gate_use_balance=True,
        gate_add_noise=True,
        gate_use_softmax=True,
    ):
        super(LinearMoELayer, self).__init__()
        assert num_selects <= num_experts  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        experts = LinearExperts(input_size, output_size, num_experts, bias=bias)
        self.gate = TopKBalancedNoisyGate(
            input_size,
            num_experts,
            num_selects,
            gate_network=gate_network,
            use_balance=gate_use_balance,
            add_noise=gate_add_noise,
            use_softmax=gate_use_softmax,
        )  # noisy gate
        self.calculator = UniversalCalculator(
            experts, multiply_gate_scores=gate_use_softmax
        )  # forward calculator for experts

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(-1, self.input_size)  # shape(batch_size*seq_len, input_size)

        # 计算被选出的专家及其分数，以及gate的loss
        indices, scores, gate_loss = self.gate(x)
        y = self.calculator(x, indices, scores)  # 合并各专家的计算结果

        y = y.reshape(
            batch_size, seq_len, self.output_size
        )  # shape(batch_size, seq_len, output_size)
        return y, gate_loss

    def change_num_selects(self, num_selects):
        if num_selects > self.gate.num_experts:
            raise ValueError(num_selects)
        else:
            self.gate.num_selects = num_selects

    def change_gate_use_balance(self, use_balance):
        self.gate.use_balance = use_balance

    def change_gate_add_noise(self, add_noise):
        self.gate.add_noise = add_noise

    def change_gate_use_softmax(self, use_softmax):
        self.gate.use_softmax = use_softmax


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
        gate_network="mlp",
        gate_use_balance=True,
        gate_add_noise=True,
        gate_use_softmax=True,
    ):
        super(LinearGLUMoELayer, self).__init__()
        assert num_selects <= num_experts  # 选择数量大于专家数量，报错
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
            bias=bias,
        )
        # noisy gate
        self.gate = TopKBalancedNoisyGate(
            input_size,
            num_experts,
            num_selects,
            gate_network=gate_network,
            use_balance=gate_use_balance,
            add_noise=gate_add_noise,
            use_softmax=gate_use_softmax,
        )
        # forward calculator for experts
        self.calculator = UniversalCalculator(
            experts, multiply_gate_scores=gate_use_softmax
        )

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(-1, self.input_size)  # shape(batch_size*seq_len, input_size)

        # 计算被选出的专家及其分数，以及gate的loss
        indices, scores, gate_loss = self.gate(x)
        y = self.calculator(x, indices, scores)  # 合并各专家的计算结果

        y = y.reshape(
            batch_size, seq_len, self.output_size
        )  # shape(batch_size, seq_len, output_size)
        return y, gate_loss

    def change_num_selects(self, num_selects):
        if num_selects > self.gate.num_experts:
            raise ValueError(num_selects)
        else:
            self.gate.num_selects = num_selects

    def change_gate_use_balance(self, use_balance):
        self.gate.use_balance = use_balance

    def change_gate_add_noise(self, add_noise):
        self.gate.add_noise = add_noise

    def change_gate_use_softmax(self, use_softmax):
        self.gate.use_softmax = use_softmax
