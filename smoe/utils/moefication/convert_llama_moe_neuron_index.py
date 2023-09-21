"""Convert a vanilla llama to llama-moe"""
import os
import shutil

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel

from smoe.models.llama_moe import (
    LlamaMoEConfig,
    LlamaMoEForCausalLM,
    LlamaMoEForSequenceClassification,
    LlamaMoEModel,
)
from smoe.utils.io import torch_load_template_file


def convert_llama_model_neuron_index(
        llama_model_path,
        split_index_path,
        select_gate_path,
        save_path,
        template,
        num_experts,
        num_selects,
        score_scale_factor=None,
        use_default_gate=False,
):
    """
    LlamaMoEModel
    """

    moe_neuron_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaModel.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        assert num_experts == len(this_layer_index)
        moe_neuron_indices.append(
            [
                torch.tensor(this_layer_index[j], dtype=torch.int)
                for j in range(num_experts)
            ]
        )

        this_layer_size_expert = [
            moe_neuron_indices[i][j].size(0) for j in range(num_experts)
        ]
        size_experts.append(this_layer_size_expert)

        if not use_default_gate:
            this_layer_gate = torch_load_template_file(select_gate_path, template, i)
            moe_gates.append(this_layer_gate)

    """build config"""
    print("Buiding llama-moe config...")
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.intermediate_size = sum(size_experts[0])
    config_llama_moe.gates = "mlp"
    config_llama_moe.score_scale_factor = 1.0 if score_scale_factor is not None else score_scale_factor

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEModel(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[1])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().half()
                elif "up" in key:
                    model_llama_moe_state_dict["layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().half()
                elif "down" in key:
                    model_llama_moe_state_dict["layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_llama_state_dict[key].transpose(0, 1)[moe_neuron_indices[layer_index][expert_index]].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        if not use_default_gate:
            model_llama_moe_state_dict["layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.0.weight"].cpu().half()
            model_llama_moe_state_dict["layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.2.weight"].cpu().half()
        model_llama_moe_state_dict["layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_llama_moe.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEModel saved to "{save_path}".')


def convert_llama_model_for_causal_lm_neuron_index(
        llama_model_path,
        split_index_path,
        select_gate_path,
        save_path,
        template,
        num_experts,
        num_selects,
        score_scale_factor=None,
        use_default_gate=False,
):
    """
    LlamaMoEForCausalLM
    """

    moe_neuron_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaForCausalLM.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        assert num_experts == len(this_layer_index)
        moe_neuron_indices.append(
            [
                torch.tensor(this_layer_index[j], dtype=torch.int)
                for j in range(num_experts)
            ]
        )

        this_layer_size_expert = [
            moe_neuron_indices[i][j].size(0) for j in range(num_experts)
        ]
        size_experts.append(this_layer_size_expert)

        if not use_default_gate:
            this_layer_gate = torch_load_template_file(select_gate_path, template, i)
            moe_gates.append(this_layer_gate)

    """build config"""
    print("Buiding llama-moe config...")
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.intermediate_size = sum(size_experts[0])
    config_llama_moe.gates = "mlp"
    config_llama_moe.score_scale_factor = 1.0 if score_scale_factor is not None else score_scale_factor

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEForCausalLM(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().half()
                elif "up" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().half()
                elif "down" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_llama_state_dict[key].transpose(0, 1)[moe_neuron_indices[layer_index][expert_index]].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        if not use_default_gate:
            model_llama_moe_state_dict["model.layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.0.weight"].cpu().half()
            model_llama_moe_state_dict["model.layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.2.weight"].cpu().half()
        model_llama_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
        # print(model_llama_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)])
    # fmt: on

    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_llama_moe.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEForCausalLM saved to "{save_path}".')


def convert_llama_model_for_sequence_classification_neuron_index(
        llama_model_path,
        split_index_path,
        select_gate_path,
        save_path,
        template,
        num_experts,
        num_selects,
        score_scale_factor=None,
        use_default_gate=False,
):
    """
    LlamaMoEForSequenceClassification
    """

    moe_neuron_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaForSequenceClassification.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        assert num_experts == len(this_layer_index)
        moe_neuron_indices.append(
            [
                torch.tensor(this_layer_index[j], dtype=torch.int)
                for j in range(num_experts)
            ]
        )

        this_layer_size_expert = [
            moe_neuron_indices[i][j].size(0) for j in range(num_experts)
        ]
        size_experts.append(this_layer_size_expert)

        if not use_default_gate:
            this_layer_gate = torch_load_template_file(select_gate_path, template, i)
            moe_gates.append(this_layer_gate)

    """build config"""
    print("Buiding llama-moe config...")
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.intermediate_size = sum(size_experts[0])
    config_llama_moe.gates = "mlp"
    config_llama_moe.score_scale_factor = 1.0 if score_scale_factor is not None else score_scale_factor

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEForSequenceClassification(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().half()
                elif "up" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().half()
                elif "down" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_llama_state_dict[key].transpose(0, 1)[moe_neuron_indices[layer_index][expert_index]].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        if not use_default_gate:
            model_llama_moe_state_dict["model.layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.0.weight"].cpu().half()
            model_llama_moe_state_dict["model.layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.2.weight"].cpu().half()
        model_llama_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_llama_moe.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEForSequenceClassification saved to "{save_path}".')


if __name__ == "__main__":
    llama_model_path = "/home/data/models/llama-transformers/7B/"
    split_index_path = "/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering/"  # split
    select_gate_path = ""  # select
    save_path = "/home/data/models/llama-moe-transformers/7B/"
    template = "layers.{}.mlp.gate_proj.weight"
    num_experts = 8
    num_selects = 2
    score_scale_factor = 8.0
    use_default_gate = True

    convert_llama_model_neuron_index(
        llama_model_path,
        split_index_path,
        select_gate_path,
        save_path,
        template,
        num_experts,
        num_selects,
        score_scale_factor=score_scale_factor,
        use_default_gate=use_default_gate,
    )

    # load test
    model_llama_moe = LlamaMoEForCausalLM.from_pretrained(save_path)
    print(model_llama_moe)
