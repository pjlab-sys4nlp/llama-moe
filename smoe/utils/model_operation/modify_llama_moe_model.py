import types

import torch

from smoe.models.llama_moe import LlamaMoEDecoderLayer, LlamaMoEModel
from smoe.modules.moe.moe_calculators import UniversalCalculator
from smoe.modules.moe.moe_gates import TopKBalancedNoisyGate
from smoe.utils.model_operation.change_llama_moe_forward import (
    forward_linear_glu_moe_layer_with_padding_mask,
    forward_llama_moe_decoder_with_hidden_states_scale_recording,
    forward_llama_moe_decoder_with_padding_mask,
    forward_llama_moe_model_with_early_stop,
    forward_llama_moe_model_with_padding_mask,
    forward_topk_balanced_noisy_gate_with_fixed_expert_selection,
    forward_topk_balanced_noisy_gate_with_hidden_states_recording,
    forward_topk_balanced_noisy_gate_with_random_expert_selection,
    forward_universal_calculator_with_scaled_gate_score,
)


def llama_moe_with_random_expert_selection(model):
    """专家每次都会随机选择n个"""
    # fmt: off
    assert isinstance(model, LlamaMoEModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer.mlp.gate, TopKBalancedNoisyGate)
        layer.mlp.gate.forward = types.MethodType(forward_topk_balanced_noisy_gate_with_random_expert_selection, layer.mlp.gate)  # change forward function for TopKBalancedNoisyGate

    return model
    # fmt: on


def llama_moe_with_fixed_expert_selection(model):
    """专家只会选择按照顺序排列的前n个"""
    # fmt: off
    assert isinstance(model, LlamaMoEModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer.mlp.gate, TopKBalancedNoisyGate)
        layer.mlp.gate.forward = types.MethodType(forward_topk_balanced_noisy_gate_with_fixed_expert_selection, layer.mlp.gate)  # change forward function for TopKBalancedNoisyGate

    return model
    # fmt: on


def llama_moe_with_hidden_states_scale_recording_early_stop(
    model, early_stop_layer=None
):
    """
    记录所有moe decoder layer中MLP的输出值大小规模，与相应的残差大小规模
    在forward时，于指定的decoder layer进行early stop
    """
    # fmt: off
    assert isinstance(model, LlamaMoEModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer, LlamaMoEDecoderLayer)

        layer.forward = types.MethodType(forward_llama_moe_decoder_with_hidden_states_scale_recording, layer)  # change forward function for LlamaMoEDecoderLayer

        layer.mlp_outputs = []
        layer.mlp_residuals = []

    model.forward = types.MethodType(forward_llama_moe_model_with_early_stop, model)  # change forward function for LlamaModel
    model.early_stop_layer = early_stop_layer

    return model
    # fmt: on


def llama_moe_with_hidden_states_scale_recording(model):
    """记录所有moe decoder layer中MLP的输出值大小规模，与相应的残差大小规模"""
    # fmt: off
    assert isinstance(model, LlamaMoEModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer, LlamaMoEDecoderLayer)

        layer.forward = types.MethodType(forward_llama_moe_decoder_with_hidden_states_scale_recording, layer)  # change forward function for LlamaMoEDecoderLayer

        layer.mlp_outputs = []
        layer.mlp_residuals = []

    return model
    # fmt: on


def llama_moe_with_scaled_gate_score(model, output_scale_factor):
    """专家的输出在乘gate score后，会根据专家数量进行缩放，之后再进行加和"""
    # fmt: off
    assert isinstance(model, LlamaMoEModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer.mlp.calculator, UniversalCalculator)

        layer.mlp.calculator.output_scale_factor = output_scale_factor
        layer.mlp.calculator.forward = types.MethodType(forward_universal_calculator_with_scaled_gate_score, layer.mlp.calculator)  # change forward function for UniversalCalculator

    return model
    # fmt: on


def llama_moe_with_hidden_states_recording(model, device):
    """记录gate的load、importance、balance_loss"""
    # fmt: off
    assert isinstance(model, LlamaMoEModel)

    model.forward = types.MethodType(forward_llama_moe_model_with_padding_mask, model)  # change forward function for LlamaMoEModel

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer.mlp.gate, TopKBalancedNoisyGate)

        layer.forward = types.MethodType(forward_llama_moe_decoder_with_padding_mask, layer)  # change forward function for LlamaMoEDecoderLayer
        layer.mlp.forward = types.MethodType(forward_linear_glu_moe_layer_with_padding_mask, layer.mlp)  # change forward function for LinearGLUMoELayer
        layer.mlp.gate.forward = types.MethodType(forward_topk_balanced_noisy_gate_with_hidden_states_recording, layer.mlp.gate)  # change forward function TopKBalancedNoisyGate

        layer.mlp.gate.samples_cnt = 0
        layer.mlp.gate.importance_sum = torch.zeros((model.config.num_experts,), device=device)
        layer.mlp.gate.importance_loss_sum = torch.zeros((1,), device=device)
        layer.mlp.gate.load_sum = torch.zeros((model.config.num_experts,), device=device)
        layer.mlp.gate.load_loss_sum = torch.zeros((1,), device=device)

    return model
    # fmt: on
