import os
import types

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
)

from smoe.utils.model_operation.change_llama_forward import (
    forward_llama_decoder_with_hidden_states_scale_recording,
    forward_llama_decoder_with_padding_mask,
    forward_llama_mlp_with_feature_dumping,
    forward_llama_model_with_padding_mask,
)


def llama_with_hidden_states_scale_recording(model):
    """记录所有decoder layer中MLP的输出值大小规模，与相应的残差大小规模"""
    # fmt: off
    assert isinstance(model, LlamaModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        assert isinstance(layer, LlamaDecoderLayer)

        layer.forward = types.MethodType(forward_llama_decoder_with_hidden_states_scale_recording, layer)  # change forward function for LlamaDecoderLayer

        layer.mlp_outputs = []
        layer.mlp_residuals = []

    return model
    # fmt: on


def llama_with_feature_dumping(model, device_id, save_path, template, save_interval=1):
    """自动保存MLP的隐层特征到文件"""
    # fmt: off
    assert isinstance(model, LlamaModel)

    model.forward = types.MethodType(forward_llama_model_with_padding_mask, model)  # change forward function for LlamaModel

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        mlp = layer.mlp
        assert isinstance(mlp, LlamaMLP)

        layer.forward = types.MethodType(forward_llama_decoder_with_padding_mask, layer)  # change forward function for LlamaDecoderLayer
        mlp.forward = types.MethodType(forward_llama_mlp_with_feature_dumping, mlp)  # change forward function for LlamaMLP

        mlp.hidden_inputs = []
        mlp.hidden_outputs = []

        mlp.device_id = device_id
        mlp.template = template
        mlp.layer_idx = layer_idx
        mlp.now_epoch = -1
        mlp.hidden_dim = model.config.hidden_size
        mlp.hidden_neurons = model.config.intermediate_size
        mlp.save_path_hidden_inputs = os.path.join(save_path, "hidden_inputs", "layer" + str(layer_idx))
        if "gate_proj" in template:
            mlp.save_path_hidden_outputs = os.path.join(save_path, "hidden_gate_outputs", "layer" + str(layer_idx))
        elif "up_proj" in template:
            mlp.save_path_hidden_outputs = os.path.join(save_path, "hidden_up_outputs", "layer" + str(layer_idx))
        mlp.save_interval = save_interval

        if not os.path.exists(mlp.save_path_hidden_inputs):
            os.makedirs(mlp.save_path_hidden_inputs)
        if not os.path.exists(mlp.save_path_hidden_outputs):
            os.makedirs(mlp.save_path_hidden_outputs)

    return model
    # fmt: on
