from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.utils import ModelOutput

from smoe.modules.moe.moe_experts import LinearGLUExperts
from smoe.modules.norm import WeightNorm
from smoe.utils.debugging import remote_breakpoint


@dataclass
class CalculatorOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None


class BaseCalculator(nn.Module):
    def __init__(self):
        super(BaseCalculator, self).__init__()

    def reset_experts(self):
        self.experts.reset_parameters()


class UniformCalculator(BaseCalculator):
    # Efficient calculator for all-select-all gates

    def __init__(self, experts, multiply_gate_scores=True, score_scale_factor=1.0):
        super(UniformCalculator, self).__init__()
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts

    def forward(self, x, topK_scores, **kwargs) -> CalculatorOutput:
        # num_selects * (bsz*seq_len, hidden_size)
        expert_outputs = [self.experts(x, i) for i in range(self.num_experts)]

        # (num_selects, bsz*seq_len, hidden_size)
        stack_expert_outputs = torch.stack(
            expert_outputs, 0
        )  # Concatenate expert outputs
        if self.multiply_gate_scores:
            expanded_socre = (
                topK_scores.transpose(0, 1)
                .unsqueeze(2)
                .expand(stack_expert_outputs.shape)
            )
            stack_expert_outputs = stack_expert_outputs * (
                expanded_socre * self.score_scale_factor
            )
        y = torch.sum(stack_expert_outputs, dim=0)

        return CalculatorOutput(hidden_states=y, num_dropped_tokens=torch.tensor(-1.0))


class UniversalCalculator(BaseCalculator):
    # Traditional calculation mode, forward $num_experts$ times with re-batch optimization
    """
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    Accepts topK scores as the dispatcher. Compared with the original SparseDispatcher, it optimizes calculations.
    The principle is still to reallocate batches for each expert.
    """

    def __init__(
        self,
        experts: LinearGLUExperts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        add_weight_norm: bool = False,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        # TODO (zhutong): use vmap to boost the training efficiency
        # self.experts_vmap = torch.vmap(self.experts)
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.mlp_norm = None
        if multiply_gate_scores and add_weight_norm:
            self.mlp_norm = WeightNorm(1, scale=score_scale_factor)
            self.mlp_norm.reset_parameters()

    def forward(
        self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs
    ) -> CalculatorOutput:
        """Forward propagation"""
        """Temporary variables"""
        batch_size = topK_indices.size(0)  # topK_indices: (bsz*seq_len, num_selects)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        # Batch indices corresponding to selected expert indices
        batch_indices = torch.arange(
            batch_size, device=topK_scores.device
        ).repeat_interleave(num_selects)

        """Generate expert indices in ascending order"""
        # index_sorted_topK_indices: (token_num*num_selects)
        _, index_sorted_topK_indices = topK_indices.sort(0)

        """Reorder scores and batch_indices according to the indices and calculate the batch size for each expert"""
        sorted_topK_scores = topK_scores.index_select(
            0, index_sorted_topK_indices
        )  # Weights corresponding to each output
        sorted_batch_indices = batch_indices.index_select(
            0, index_sorted_topK_indices
        )  # Token indices in each batch for each expert

        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount(
                minlength=self.num_experts
            ).tolist()  # Batch size for each expert

        """Reorganize batches for each expert"""
        sorted_x = x.index_select(
            0, sorted_batch_indices
        )  # Reorganize inputs based on batch indices
        split_x = torch.split(
            sorted_x, expert_batch_size, dim=0
        )  # Divide inputs based on expert batch sizes

        """Forward propagation for each expert"""  # Potential for parallel optimization here
        expert_outputs = [
            self.experts(split_x[i], i)
            for i in range(self.num_experts)
            if split_x[i].shape[0] > 0
        ]

        """Reassemble and weight the outputs from each expert"""
        cat_expert_outputs = torch.cat(expert_outputs, 0)  # Concatenate expert outputs
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor,
                )
            else:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs, sorted_topK_scores.reshape(-1, 1)
                )
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        zeros = torch.zeros(
            (batch_size, output_dim),
            device=cat_expert_outputs.device,
            dtype=cat_expert_outputs.dtype,
        )
        y = zeros.index_add(
            0, sorted_batch_indices, cat_expert_outputs
        )  # Add outputs based on batch indices

        return CalculatorOutput(hidden_states=y, num_dropped_tokens=torch.tensor(-1.0))


class SwitchDropTokenCalculator(BaseCalculator):
    """
    https://arxiv.org/pdf/2101.03961.pdf
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
    Calculator with capacity_factor, automatically drops tokens exceeding capacity
    """

    def __init__(
        self,
        experts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        drop_tokens=True,
        dropped_padding="zero",  # Zero input
        capacity_factor=1.25,
        add_weight_norm: bool = False,
    ):
        super(SwitchDropTokenCalculator, self).__init__()
        self.available_dropped_padding_choices = ("zero", "input")
        assert dropped_padding in self.available_dropped_padding_choices
        # Ensure input and output dimensions match if dropping tokens
        if drop_tokens and dropped_padding != "zero":
            assert experts.in_features == experts.out_features

        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.out_features = experts.out_features
        self.mlp_norm = None
        if multiply_gate_scores and add_weight_norm:
            self.mlp_norm = WeightNorm(
                self.experts.out_features, scale=score_scale_factor
            )
            self.mlp_norm.reset_parameters()

        # Capacity
        self.drop_tokens = drop_tokens
        self.dropped_padding = dropped_padding
        self.capacity_factor = capacity_factor

    def forward(self, x, topK_indices, topK_scores, **kwargs) -> CalculatorOutput:
        """
        Args:
            x: (bsz*seq_len, hidden_size) bsz*seq_len is the total number of tokens in this batch
            topK_indices: (bsz*seq_len,) each element represents the expert idx to consume the token
                e.g. topK_indices[1] = 3 means the token-1 is assigned to expert-3
        """
        batch_size = topK_indices.size(0)
        capacity = int(self.capacity_factor * batch_size / self.num_experts)
        dropped_indices = []
        y = torch.zeros((batch_size, self.out_features), device=x.device, dtype=x.dtype)

        # Forward propagation for each expert. There is potential for parallel optimization here.
        num_dropped_tokens = -1
        for i in range(self.num_experts):
            # token_indices is a tensor of (num_tokens_in_this_expert,)
            #   where each element denotes the token position idx
            token_indices = (topK_indices == i).nonzero(as_tuple=True)[0]
            num_assigned_tokens = token_indices.numel()
            # Ignore if the expert is not over capacity
            if self.drop_tokens and num_assigned_tokens > capacity:
                shuffled_indices = torch.randperm(num_assigned_tokens, device=x.device)
                # Shuffle indexes before dropping
                token_indices = token_indices[shuffled_indices]
                # Collect the tokens over capacity as dropped tokens
                dropped_indices.append(token_indices[capacity:])
                # Keep only the tokens upto the capacity of the expert
                token_indices = token_indices[:capacity]
                num_dropped_tokens = num_assigned_tokens - capacity

            if num_assigned_tokens > 0:
                expert_output = self.experts(x[token_indices, :], i)
                y[token_indices, :] = expert_output

        if self.dropped_padding == "input" and len(dropped_indices) > 0:
            dropped_indices = torch.cat(dropped_indices, dim=0)
            y[dropped_indices, :] = x[dropped_indices, :]

        if self.multiply_gate_scores:
            y = torch.mul(y, topK_scores.reshape(-1, 1) * self.score_scale_factor)
            if self.mlp_norm is not None:
                y = self.mlp_norm(y)

        return CalculatorOutput(
            hidden_states=y, num_dropped_tokens=torch.tensor(num_dropped_tokens)
        )
