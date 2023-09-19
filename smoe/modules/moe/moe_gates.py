import math

import torch
from deepspeed.moe.sharded_moe import gumbel_rsample
from einops import einsum, rearrange
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
        # self.mean = torch.tensor([0.0], requires_grad=False)
        # self.std = torch.tensor([1.0], requires_grad=False)
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

        """专家平衡选择"""
        # zhutong: 不要把`self.training`写在里面的if语句中，否则会导致eval模式下gate loss输出值设备不匹配的错误
        if self.training and self.use_balance:
            """计算importance"""
            zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
            scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
            importance = scores_filtered.sum(0)  # shape(num_experts)

            """计算load"""
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

            """计算balance loss"""
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight

        else:
            balance_loss = None

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load.tolist(),
            "importance": importance.tolist(),
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

        """专家平衡选择"""
        if self.use_balance:
            """计算importance"""
            zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
            scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
            importance = scores_filtered.sum(0)  # shape(num_experts)

            """计算load"""
            if self.training:  # 计算各分数在处于topK范围内的概率，可传递梯度
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
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight

        else:
            balance_loss = None

        return {
            "scores": scores,
            "balance_loss": balance_loss,
        }
    # fmt: on

    def reset_gate_network(self):
        for name, param in self.gate_network.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)


class SwitchBalancedGate(nn.Module):
    """
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
        balance_loss_weight=1e-1,
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

        load = top1_indices.bincount(minlength=self.num_experts)
        assert load.shape[0] == self.num_experts
        # load = top1_indices.bincount()  # 不传递梯度，与原论文保持一致
        # if load.shape[0] < self.num_experts:  # 如果长度不足，则使用0补齐load矩阵
        #     pad_tensor = torch.zeros((self.num_experts - load.shape[0],), device=load.device, dtype=torch.int).flatten()
        #     load = torch.cat((load, pad_tensor), dim=0)
        # print(f"ZHUTONG (RANK: {os.environ['RANK']}): GATE FORWARD LOAD: {load=}")
        load_mean = load / batch_size  # shape(num_experts)

        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)
        balance_loss *= self.balance_loss_weight

        return {
            "topK_indices": top1_indices,
            "topK_scores": top1_scores,
            "expert_batch_size": load.tolist(),
            "balance_loss": balance_loss,
            "load": load_mean.tolist(),
            "importance": importance_mean.tolist(),
        }

    def reset_gate_network(self):
        for name, param in self.gate_network.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)


class SoftMoEGate(nn.Module):
    """
    https://arxiv.org/pdf/2308.00951.pdf
    https://github.com/fkodom/soft-mixture-of-experts
    """

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        slots_per_expert: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert

        self.phi = nn.Parameter(
            torch.empty(
                (in_features, num_experts, slots_per_expert),
                device=device,
                dtype=dtype,
            )
        )

    def reset_gate_network(self) -> None:
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x):
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected x.size(-1)={x.size(-1)} to match embed_dim={self.in_features}, "
                f"but got {x.size(-1)}."
            )
        elif x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions, but got {x.ndim}.")

        logits = einsum(x, self.phi, "b m d, d n p -> b m n p")
        dispatch_weights = logits.softmax(dim=0)  # denoted 'D' in the paper
        # NOTE: The 'torch.softmax' function does not support multiple values for the
        # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # Then, we rearrange the Tensor into its original shape.
        combine_weights = rearrange(
            logits.flatten(start_dim=2).softmax(dim=-1),
            "b m (n p) -> b m n p",
            n=self.num_experts,
        )

        return {
            "dispatch_weights": dispatch_weights,
            "combine_weights": combine_weights,
            "balance_loss": torch.tensor(0),
        }
