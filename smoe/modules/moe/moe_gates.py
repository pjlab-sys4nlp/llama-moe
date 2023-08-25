import torch
from torch import nn
from torch.distributions.normal import Normal

valid_gate_type = ("linear", "mlp")


def get_gate_network(gate_type, input_size, num_experts):
    gate_type = gate_type.lower()

    if gate_type == "linear":
        gate_network = nn.Linear(input_size, num_experts, bias=False)
        nn.init.zeros_(gate_network.weight)
    elif gate_type == "mlp":
        gate_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_experts, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(num_experts, num_experts, bias=False),
        )
    else:
        raise ValueError('Expected "gate_type" in', valid_gate_type, "got", gate_type)

    return gate_network


class TopKBalancedNoisyGate(nn.Module):
    """
    https://arxiv.org/abs/1701.06538.
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_balance=True,
        add_noise=True,
        use_softmax=True,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.use_balance = use_balance
        self.add_noise = add_noise
        self.use_softmax = use_softmax

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        # add_noise
        self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
        self.weight_noise.weight.data = torch.zeros(
            (num_experts, input_size),
            requires_grad=True,
            device=self.weight_noise.weight.data.device,
            dtype=self.weight_noise.weight.data.dtype,
        )
        # print(self.weight_noise.weight.data)
        self.mean = torch.tensor([0.0], device=self.weight_noise.weight.data.device)
        self.std = torch.tensor([1.0], device=self.weight_noise.weight.data.device)
        self.normal = Normal(self.mean, self.std)
        self.softplus = nn.Softplus()

        # use_softmax
        self.softmax = nn.Softmax(1)

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    # fmt: off
    def forward(self, x, noise_epsilon=1e-2, gate_loss_lambda=1e-2):
        """先计算所有专家的权重值"""
        logits_gate = self.gate_network(x)  # gate计算出的权重
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)  # 噪声矩阵计算结果
            noise_control = self.softplus(noise_mm) + noise_epsilon  # 控制器得到的噪声增加量
            logits_noise = torch.randn_like(logits_gate) * noise_control  # noise附加的权重
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate  # 最终权重，shape(batch_size, num_experts)

        """选出前k个权重，并计算各个专家的分数scores"""
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]

        if self.use_softmax:
            top_k_scores = self.softmax(top_k_logits)  # 对前k个计算softmax，得到对应的分数
        else:
            top_k_scores = top_k_logits

        """专家平衡选择"""
        if self.use_balance:

            """计算importance"""
            zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
            scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
            importance = scores_filtered.sum(0)  # shape(num_experts)

            """计算load"""
            if self.training and self.add_noise:  # 计算各分数在给定随机噪声的情况下，处于topK范围内的概率
                batch_size = logits_gate.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
                threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = self.normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)  # shape(num_experts)

            """计算balance loss"""
            gate_loss = self.cv_squared(importance) + self.cv_squared(load)
            gate_loss *= gate_loss_lambda

        else:
            gate_loss = None

        return top_k_indices, top_k_scores, gate_loss

    def forward_return_scores(self, x, noise_epsilon=1e-2, gate_loss_lambda=1e-2):
        """先计算所有专家的权重值"""
        logits_gate = self.gate_network(x)  # gate计算出的权重
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)  # 噪声矩阵计算结果
            noise_control = self.softplus(noise_mm) + noise_epsilon  # 控制器得到的噪声增加量
            logits_noise = torch.randn_like(logits_gate) * noise_control  # noise附加的权重
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate  # 最终权重

        """计算各个专家的分数scores"""
        if self.use_softmax:
            scores = self.softmax(logits)
        else:
            scores = logits

        """选出前k个权重，并计算各个专家的分数scores"""
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]

        if self.use_softmax:
            top_k_scores = self.softmax(top_k_logits)  # 对前k个计算softmax，得到对应的分数
        else:
            top_k_scores = top_k_logits

        """专家平衡选择"""
        if self.training and self.use_balance:

            """计算importance"""
            zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
            scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
            importance = scores_filtered.sum(0)  # shape(num_experts)

            """计算load"""
            if self.add_noise:  # 计算各分数在给定随机噪声的情况下，处于topK范围内的概率
                batch_size = logits_gate.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()

                if self.mean.device != x.device:
                    self.mean = self.mean.to(x.device)
                    self.std = self.std.to(x.device)
                normal = Normal(self.mean, self.std)

                threshold_positions_if_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
                threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
                # is each value currently in the top k.
                prob_if_in = normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)  # shape(num_experts)

            """计算balance loss"""
            gate_loss = self.cv_squared(importance) + self.cv_squared(load)
            gate_loss *= gate_loss_lambda

        else:
            gate_loss = None

        return scores, gate_loss

    # fmt: on

    def reinit_gate_network(self):
        for name, param in self.gate_network.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)
