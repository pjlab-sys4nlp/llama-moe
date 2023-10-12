import warnings

import torch
from deepspeed.moe.sharded_moe import gumbel_rsample
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
        raise ValueError(f'Expected "gate_type" in {valid_gate_type}, got {gate_type}.')

    return gate_network


class BaseGate(nn.Module):
    def __init__(self):
        super(BaseGate, self).__init__()

    def reset_gate_network(self):
        if "gate_network_type" not in vars(self):
            raise KeyError(f"{type(self)} does not have a gate network.")
        else:
            self.gate_network = get_gate_network(
                self.gate_network_type, self.input_size, self.num_experts
            )


class UniformPlainGate(BaseGate):
    """
    Select all experts with the same score.
    If use_softmax=True, then score=1/num_experts.
    If use_softmax=False, then score=1.
    """

    def __init__(
        self,
        input_size,
        num_experts,
        use_softmax=True,
    ):
        super(UniformPlainGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.use_softmax = use_softmax

    def forward(self, x):
        batch_size = x.shape[0]  # gate计算出的权重
        scores = torch.ones((batch_size, self.num_experts), device=x.device)
        if self.use_softmax:
            scores /= self.num_experts
        indices = (
            torch.arange(0, self.num_experts, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, self.num_experts)
        )

        return {
            "topK_indices": indices,
            "topK_scores": scores,
            "balance_loss": torch.tensor(0, device=x.device),
            "load": torch.tensor(-1, device=x.device),
            "importance": torch.tensor(-1, device=x.device),
        }


class RandomLearnableGate(BaseGate):
    """
    Randomly select k experts each time, with a learnable gate_network controlling expert scores.
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(RandomLearnableGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon

    def forward(self, x):
        logits = self.gate_network(x)  # gate计算出的权重
        gumbel_rsample(logits.shape, device=logits.device).to(
            logits
        ) * self.noise_epsilon

        _, top_k_indices = torch.rand_like(logits).topk(self.num_selects, dim=1)
        top_k_logits = torch.gather(logits, dim=1, index=top_k_indices)
        top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": torch.tensor(0, device=x.device),
            "load": torch.tensor(-1, device=x.device),
            "importance": torch.tensor(-1, device=x.device),
        }


class TopKBalancedNoisyGate(BaseGate):
    """
    Select the top-k experts each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/abs/1701.06538.
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
        self.weight_noise.weight.data = torch.zeros(
            (num_experts, input_size),
            requires_grad=True,
            device=self.weight_noise.weight.data.device,
            dtype=self.weight_noise.weight.data.dtype,
        )
        # print(self.weight_noise.weight.data)
        self.mean = 0.0
        self.std = 1.0
        self.normal = Normal(self.mean, self.std)
        self.softplus = nn.Softplus()

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
    def forward(self, x):
        """先计算所有专家的权重值"""
        logits_gate = self.gate_network(x)  # gate计算出的权重
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)  # 噪声矩阵计算结果
            noise_control = self.softplus(noise_mm) + self.noise_epsilon  # 控制器得到的噪声增加量
            logits_noise = torch.randn_like(logits_gate) * noise_control  # noise附加的权重
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate  # 最终权重，shape(batch_size, num_experts)

        """选出前k个权重，并计算各个专家的分数scores"""
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]
        top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits

        """计算importance"""
        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        """计算load"""
        # zhutong: 不要把`self.training`写在里面的if语句中，否则会导致eval模式下balance_loss输出值设备不匹配的错误
        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
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
                load = (scores_filtered > 0).sum(0)
                warnings.warn('Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                              'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.')
        else:
            load = (scores_filtered > 0).sum(0)

        """计算balance loss"""
        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(0, device=x.device)

        # print("weight", self.gate_network.weight, sep="\n")
        # print("logits_gate", logits_gate, sep="\n")
        # print("importance", importance, sep="\n")
        # print("load", load, sep="\n")
        # print("balance_loss", balance_loss, sep="\n")

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

    def forward_return_scores(self, x):
        logits_gate = self.gate_network(x)  # gate计算出的权重
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)  # 噪声矩阵计算结果
            noise_control = self.softplus(noise_mm) + self.noise_epsilon  # 控制器得到的噪声增加量
            logits_noise = torch.randn_like(logits_gate) * noise_control  # noise附加的权重
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate  # 最终权重

        """计算各个专家的分数scores"""
        scores = self.softmax(logits) if self.use_softmax else logits

        """选出前k个权重，并计算各个专家的分数scores"""
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]
        top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits

        """计算importance"""
        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        """计算load"""
        # zhutong: 不要把`self.training`写在里面的if语句中，否则会导致eval模式下balance_loss输出值设备不匹配的错误
        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
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
                load = (scores_filtered > 0).sum(0)
                warnings.warn("Gradient-trackable implementation for load calculation is only available when \"add_noise=True\". "
                              "Training without noise will block the gradient from load path and lead to inconsistency in optimization objective.")
        else:
            load = (scores_filtered > 0).sum(0)

        """计算balance loss"""
        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = None

        return {
            "scores": scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

    # fmt: on


class SwitchBalancedGate(BaseGate):
    """
    Select 1 expert each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/pdf/2101.03961.pdf
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
    ):
        super(SwitchBalancedGate, self).__init__()
        assert num_selects == 1
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight
        self.add_noise = add_noise

    # fmt: off
    def forward(self, x):
        batch_size = x.shape[0]
        logits = self.gate_network(x)  # shape(batch_size, num_experts)
        scores = self.softmax(logits) if self.use_softmax else logits
        if self.add_noise:
            # .to(logits) to make sure the noise is in the same dtype as logits
            #   (e.g. bfloat16) while the default type is float32
            logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device).to(logits)
        else:
            logits_w_noise = logits
        top1_scores, top1_indices = torch.max(logits_w_noise, dim=1)

        """balance loss"""
        importance_mean = scores.mean(0)  # shape(num_experts)

        load = top1_indices.bincount(minlength=self.num_experts)  # 不传递梯度，与原论文保持一致
        assert load.shape[0] == self.num_experts
        # print(f"ZHUTONG (RANK: {os.environ['RANK']}): GATE FORWARD LOAD: {load=}")
        load_mean = load / batch_size  # shape(num_experts)

        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)
        balance_loss *= self.balance_loss_weight

        return {
            "topK_indices": top1_indices,
            "topK_scores": top1_scores,
            "expert_batch_size": load.tolist(),
            "balance_loss": balance_loss,
            "load": load_mean,
            "importance": importance_mean,
        }
