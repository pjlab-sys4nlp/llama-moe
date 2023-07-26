"""Convert a vanilla llama to llama-moe"""
import os
import shutil
import sys
from collections import Counter

import torch
from llama_moe import LlamaMoEConfig, LlamaMoEForCausalLM, LlamaMoEModel
from run_moefication.moefication_utils.expert_select import load_template_file
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaModel

# sys.path.append("/home/dongdz/workspace/moefication/")
# print(sys.path)


def convert_llama_model(
    llama_model_path,
    split_index_path,
    select_gate_path,
    save_path,
    template,
    num_experts,
    num_selects,
):
    moe_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaModel.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = load_template_file(split_index_path, template, i)
        this_layer_gate = load_template_file(select_gate_path, template, i)
        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]

        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))
        moe_gates.append(this_layer_gate)
        size_experts.append(this_layer_size_expert)

    """build config"""
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.gates = "mlp"

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEModel(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[1])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict[
                        "layers.{}.mlp.calculator.experts.weight_gate.{}".format(
                            layer_index, expert_index
                        )
                    ] = (
                        model_llama_state_dict[key][
                            moe_indices[layer_index] == expert_index
                        ]
                        .cpu()
                        .half()
                    )
                elif "up" in key:
                    model_llama_moe_state_dict[
                        "layers.{}.mlp.calculator.experts.weight_up.{}".format(
                            layer_index, expert_index
                        )
                    ] = (
                        model_llama_state_dict[key][
                            moe_indices[layer_index] == expert_index
                        ]
                        .cpu()
                        .half()
                    )
                elif "down" in key:
                    model_llama_moe_state_dict[
                        "layers.{}.mlp.calculator.experts.weight_down.{}".format(
                            layer_index, expert_index
                        )
                    ] = (
                        model_llama_state_dict[key]
                        .transpose(0, 1)[moe_indices[layer_index] == expert_index]
                        .transpose(0, 1)
                        .cpu()
                        .half()
                    )

    for layer_index in range(num_layers):
        model_llama_moe_state_dict[
            "layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)
        ] = (moe_gates[layer_index]._modules["0"].weight.cpu().half())
        model_llama_moe_state_dict[
            "layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)
        ] = (moe_gates[layer_index]._modules["2"].weight.cpu().half())

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


def convert_llama_model_for_causal_lm(
    llama_model_path,
    split_index_path,
    select_gate_path,
    save_path,
    template,
    num_experts,
    num_selects,
):
    moe_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaForCausalLM.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = load_template_file(split_index_path, template, i)
        this_layer_gate = load_template_file(select_gate_path, template, i)
        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]

        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))
        moe_gates.append(this_layer_gate)
        size_experts.append(this_layer_size_expert)

    """build config"""
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.gates = "mlp"

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEForCausalLM(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict[
                        "model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(
                            layer_index, expert_index
                        )
                    ] = (
                        model_llama_state_dict[key][
                            moe_indices[layer_index] == expert_index
                        ]
                        .cpu()
                        .half()
                    )
                elif "up" in key:
                    model_llama_moe_state_dict[
                        "model.layers.{}.mlp.calculator.experts.weight_up.{}".format(
                            layer_index, expert_index
                        )
                    ] = (
                        model_llama_state_dict[key][
                            moe_indices[layer_index] == expert_index
                        ]
                        .cpu()
                        .half()
                    )
                elif "down" in key:
                    model_llama_moe_state_dict[
                        "model.layers.{}.mlp.calculator.experts.weight_down.{}".format(
                            layer_index, expert_index
                        )
                    ] = (
                        model_llama_state_dict[key]
                        .transpose(0, 1)[moe_indices[layer_index] == expert_index]
                        .transpose(0, 1)
                        .cpu()
                        .half()
                    )

    for layer_index in range(num_layers):
        model_llama_moe_state_dict[
            "model.layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)
        ] = (moe_gates[layer_index]._modules["0"].weight.cpu().half())
        model_llama_moe_state_dict[
            "model.layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)
        ] = (moe_gates[layer_index]._modules["2"].weight.cpu().half())

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


if __name__ == "__main__":
    llama_model_path = "/home/data/models/llama-transformers/7B/"
    split_index_path = "/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering/"  # split
    select_gate_path = "/home/dongdz/workspace/moefication/llama_moe_temp_files/7B-8Expert-Select-MLP/"  # select
    save_path = "/home/data/models/llama-moe-transformers/7B/"
    template = "layers.{}.mlp.gate_proj.weight"
    num_experts = 8
    num_selects = 2

    convert_llama_model(
        llama_model_path,
        split_index_path,
        select_gate_path,
        save_path,
        template,
        num_experts,
        num_selects,
    )

    # load test
    model_llama_moe = LlamaMoEForCausalLM.from_pretrained(save_path)
    print(model_llama_moe)
