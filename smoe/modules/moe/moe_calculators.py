import torch
from torch import nn


class UniversalCalculator(nn.Module):
    # traditional calculation mode, forward $num_experts$ times with re-batch optimization
    """
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    接收topK scores的DisPatcher，相比原版的SparseDispatcher进行了计算上的优化
    原理依旧是重新分配各个专家的batch。
    """

    def __init__(self, experts, multiply_gate_scores=True):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.num_experts = experts.num_experts

    def forward(self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs):
        # fmt: off
        """正向传播"""
        """临时变量"""
        batch_size = topK_indices.size(0)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        batch_indices = torch.arange(batch_size, device=topK_scores.device).repeat_interleave(num_selects)  # 选出的专家编号所对应的batch编号，shape(batch_size*num_selects)

        """按照专家序号从小到大的顺序，生成专家索引"""
        _, index_sorted_topK_indices = topK_indices.sort(0)

        """按照索引重新排列scores与batch_indices，并计算专家的batch_size"""
        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)  # 各个输出对应的权重
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)  # 各个专家对应的batch编号

        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount().tolist()  # 各个专家对应的batch_size
            if len(expert_batch_size) < self.num_experts:  # 列表长度不足专家数，说明 被选择的最大专家序号 小于 所有专家中的最大专家序号
                expert_batch_size.extend([0] * (self.num_experts - len(expert_batch_size)))  # 使用0补全列表

        """对每个专家重新组合batch"""
        sorted_x = x.index_select(0, sorted_batch_indices).squeeze(1)  # 将输入按照排序后的batch编号，重新编制
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)  # 按照排序后每个专家的batch_size进行分隔，恰好得到各个专家所需的batch

        """各专家分别正向传播"""  # 此处应该有并行优化的空间 (如果单次forward不足以占满显卡利用率)
        expert_outputs = [self.experts(split_x[i], i) for i in range(self.num_experts) if split_x[i].shape[0] > 0]

        """重组各个专家的输出，并进行加权"""
        cat_expert_outputs = torch.cat(expert_outputs, 0)  # 拼接专家输出
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1))  # 乘权重
        zeros = torch.zeros((batch_size, output_dim), device=cat_expert_outputs.device, dtype=cat_expert_outputs.dtype)
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)  # 按照对应的batch编号，添加输出

        return y
        # fmt: on


class SwitchDropTokenCalculator(nn.Module):
    """
    https://arxiv.org/pdf/2101.03961.pdf
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feed_forward.py
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
    带有capacity_factor的计算器，自动丢弃超出容量的token
    """

    def __init__(
        self,
        experts,
        multiply_gate_scores=True,
        drop_tokens=True,
        capacity_factor=1.25,
    ):
        super(SwitchDropTokenCalculator, self).__init__()
        if drop_tokens:  # 如果丢弃token，则必须保证输入输出维度相同
            assert experts.in_features == experts.out_features
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.num_experts = experts.num_experts
        self.out_features = experts.out_features

        # capacity
        self.drop_tokens = drop_tokens
        self.capacity_factor = capacity_factor

    def forward(self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs):
        # fmt: off
        """正向传播"""
        """临时变量"""
        batch_size = topK_indices.size(0)
        capacity = int(self.capacity_factor * batch_size / self.num_experts)
        dropped_indices = []
        y = torch.zeros((batch_size, self.out_features), device=x.device, dtype=x.dtype)

        """计算专家的batch_size"""
        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount().tolist()  # 各个专家对应的batch_size
            if len(expert_batch_size) < self.num_experts:  # 列表长度不足专家数，说明 被选择的最大专家序号 小于 所有专家中的最大专家序号
                expert_batch_size.extend([0] * (self.num_experts - len(expert_batch_size)))  # 使用0补全列表

        """各专家分别正向传播"""  # 此处应该有并行优化的空间 (如果单次forward不足以占满显卡利用率)
        for i in range(self.num_experts):
            batch_indices = (topK_indices == i).nonzero(as_tuple=True)[0]
            if self.drop_tokens and expert_batch_size[i] > capacity:  # Ignore if the expert is not over capacity
                batch_indices = batch_indices[torch.randperm(expert_batch_size[i], device=x.device)]  # Shuffle indexes before dropping
                dropped_indices.append(batch_indices[capacity:])  # Collect the tokens over capacity as dropped tokens
                batch_indices = batch_indices[:capacity]  # Keep only the tokens upto the capacity of the expert

            if expert_batch_size[i] > 0:
                expert_output = self.experts(x[batch_indices, :], i)
                if self.multiply_gate_scores:
                    batch_scores = topK_scores[batch_indices]
                    expert_output = torch.mul(expert_output, batch_scores.reshape(-1, 1))  # 乘权重
                y[batch_indices, :] = expert_output

        if len(dropped_indices) > 0:
            dropped_indices = torch.cat(dropped_indices, dim=0)
            y[dropped_indices, :] = x[dropped_indices, :]

        return y
        # fmt: on
