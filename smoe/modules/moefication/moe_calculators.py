import torch
from torch import nn

# valid_mode = ("default", "faster")


# def get_linear_layer_calculator(mode, experts, kernel="linear", multiply_by_gates=True):
#     if mode == "default":
#         calculator = UniversalCalculator(experts, multiply_gate_scores=multiply_by_gates)
#     elif mode == "faster":
#         calculator = MoE_FasterLinearCalculator(experts, kernel=kernel, multiply_by_gates=multiply_by_gates)
#     else:
#         raise ValueError("Invalid calculation mode, expected " + str(valid_mode) + ", get " + mode + ".")
#
#     return calculator


class UniversalCalculator(
    nn.Module
):  # traditional calculation mode, forward $num_experts$ times with re-batch optimization
    """
    接收topK scores的DisPatcher，相比原版的SparseDispatcher进行了计算上的优化
    原理依旧是重新分配各个专家的batch。
    """

    def __init__(self, experts, multiply_gate_scores=True):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.num_experts = experts.num_experts

    def forward(self, x, topK_indices, topK_scores):
        """正向传播"""
        """临时变量"""
        batch_size = topK_indices.size(0)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        batch_indices = (
            torch.arange(batch_size)
            .repeat_interleave(num_selects)
            .to(topK_scores.device)
        )  # 选出的专家编号所对应的batch编号，shape(batch_size*num_selects)

        """按照专家序号从小到大的顺序，生成专家索引"""
        _, index_sorted_topK_indices = topK_indices.sort(0)

        """按照索引重新排列scores与batch_indices，并计算专家的batch_size"""
        sorted_topK_scores = topK_scores.index_select(
            0, index_sorted_topK_indices
        )  # 各个输出对应的权重
        sorted_batch_indices = batch_indices.index_select(
            0, index_sorted_topK_indices
        )  # 各个专家对应的batch编号
        expert_batch_size = topK_indices.bincount().tolist()  # 各个专家对应的batch_size
        if (
            len(expert_batch_size) < self.num_experts
        ):  # 列表长度不足专家数，说明 被选择的最大专家序号 小于 所有专家中的最大专家序号
            expert_batch_size.extend(
                [0] * (self.num_experts - len(expert_batch_size))
            )  # 使用0补全列表

        """对每个专家重新组合batch"""
        sorted_x = x.index_select(0, sorted_batch_indices).squeeze(
            1
        )  # 将输入按照排序后的batch编号，重新编制
        split_x = torch.split(
            sorted_x, expert_batch_size, dim=0
        )  # 按照排序后每个专家的batch_size进行分隔，恰好得到各个专家所需的batch
        # print(expert_batch_size)
        # print(len(split_x))

        """各专家分别正向传播"""  # 此处应该有并行优化的空间 (如果单次forward不足以占满显卡利用率)
        expert_outputs = [
            self.experts(split_x[i], i)
            for i in range(self.num_experts)
            if split_x[i].shape[0] > 0
        ]

        """重组各个专家的输出，并进行加权"""
        cat_expert_outputs = torch.cat(expert_outputs, 0)  # 拼接专家输出
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            cat_expert_outputs = torch.mul(
                cat_expert_outputs, sorted_topK_scores.reshape(-1, 1)
            )  # 乘权重
        zeros = torch.zeros(
            (batch_size, output_dim),
            requires_grad=True,
            device=cat_expert_outputs.device,
        )
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs).to(
            cat_expert_outputs.device
        )  # 按照对应的batch编号，添加输出
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return y


"""下述代码为失败方案，不要启用"""
# class MoE_FasterLinearCalculator(nn.Module): # aggregate expert weights for all inputs, forward 1 time with group-conv transformation optimization
#     """
#     使用优化方法，先合并权重，再正向传播，只需要1次计算
#     """
#
#     def __init__(self, experts, kernel="linear", multiply_by_gates=True):
#         super(MoE_FasterLinearCalculator, self).__init__()
#         self.experts = experts
#         self.kernel = get_kernel(kernel)
#         self.multiply_by_gates = multiply_by_gates
#         # 基本信息
#         self.output_dim = experts.weight.size(1)
#         self.input_dim = experts.weight.size(2)
#         print("input_dim: ", self.input_dim)
#         print("output_dim: ", self.output_dim)
#
#     def forward(self, x, topK_indices, topK_scores):
#         """正向传播"""
#         """临时变量"""
#         batch_size = topK_indices.size(0)
#         num_selects = topK_indices.size(1)
#         topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
#         topK_scores_weights = topK_scores.reshape(batch_size * num_selects, 1, 1)  # shape(batch_size*num_selects, 1, 1)
#         topK_scores_bias = topK_scores.reshape(batch_size * num_selects, 1)  # shape(batch_size*num_selects, 1)
#         print("batch_size: ", batch_size)
#
#         """先将参数通过核函数"""
#         kernel_weights = self.kernel(self.experts.weight)  # shape(num_experts, output_dim, input_dim)
#         kernel_bias = self.kernel(self.experts.bias)  # shape(num_experts, output_dim)
#         print("kernel_weights: ", kernel_weights.size())
#
#         """按照选出的专家编号，重新排列参数"""
#         sorted_weights = kernel_weights.index_select(0, topK_indices)  # shape(batch_size*num_selects, output_dim, input_dim)
#         sorted_bias = kernel_bias.index_select(0, topK_indices)  # shape(batch_size*num_selects, output_dim)
#         print("sorted_weights: ", sorted_weights.size())
#
#         """计算新的参数"""
#         # 先算新的weights参数
#         sorted_weights.mul_(topK_scores_weights)  # 权重相乘
#         sorted_weights = sorted_weights.reshape((batch_size, num_selects, self.output_dim, self.input_dim))  # 转换形状
#         sorted_weights = sorted_weights.sum(1)  # 合并参数，shape(batch_size, output_dim, input_dim)
#         print("sorted_weights after sum: ", sorted_weights.size())
#
#         # 再算新的bias参数
#         sorted_bias.mul_(topK_scores_bias)  # 权重相乘
#         sorted_bias = sorted_bias.reshape((batch_size, num_selects, self.output_dim))  # 转换形状
#         sorted_bias = sorted_bias.sum(1)  # 合并参数，shape(batch_size, output_dim)
#
#         """转换全连接为卷积形式，使用分组卷积进行并行计算"""
#         # https://zhuanlan.zhihu.com/p/208519425
#         x = x.reshape(1, batch_size, self.input_dim, 1)  # 对应卷积图像shape(batch_size, channels, height, width)
#         conv_weight = sorted_weights.reshape(batch_size * self.output_dim, 1, self.input_dim, 1)  # 对应卷积核权重shape(out_channels, in_channels, size[0], size[1])
#         conv_bias = sorted_bias.reshape(batch_size * self.output_dim)  # 对应卷积核偏置shape(out_channels)
#         y = F.conv2d(x, weight=conv_weight, bias=conv_bias, groups=batch_size)  # 分组卷积
#         y = y.reshape(batch_size, self.output_dim)
#
#         return y
