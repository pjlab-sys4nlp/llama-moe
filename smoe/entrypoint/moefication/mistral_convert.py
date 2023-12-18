import argparse
import os
import shutil

import torch
from tqdm import tqdm
from transformers import MistralForCausalLM, MixtralConfig, MixtralForCausalLM

from smoe.utils.io import torch_load_template_file


def convert(
        model_path,
        moe_config_path,
        split_index_path,
        save_path,
        num_experts,
        num_selects,
        template,
):
    """load model"""
    print("Loading model...", flush=True)
    model = MistralForCausalLM.from_pretrained(model_path)
    model.to("cpu")
    model_state_dict = model.state_dict()

    """load indices and gate weights"""
    moe_indices = []
    num_layers = model.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        print(this_layer_index, flush=True)
        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

    size_expert = torch.sum(moe_indices[0] == 0).item()

    """build config"""
    print("Buiding moe config...", flush=True)
    config_moe = MixtralConfig.from_pretrained(moe_config_path)
    config_moe.intermediate_size = size_expert
    config_moe.num_experts_per_tok = num_selects
    config_moe.num_local_experts = num_experts

    """initialize moe model"""
    print("Initializing moe model...", flush=True)
    model_moe = MixtralForCausalLM(config_moe)
    model_moe.to("cpu")
    model_moe_state_dict = model_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...", flush=True)
    for key in model_state_dict.keys():
        if "mlp" not in key:
            model_moe_state_dict[key] = model_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_moe_state_dict["model.layers.{}.block_sparse_moe.experts.{}.w1.weight".format(layer_index, expert_index)] = model_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "up" in key:
                    model_moe_state_dict["model.layers.{}.block_sparse_moe.experts.{}.w3.weight".format(layer_index, expert_index)] = model_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "down" in key:
                    model_moe_state_dict["model.layers.{}.block_sparse_moe.experts.{}.w2.weight".format(layer_index, expert_index)] = model_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()
    # fmt: on

    print("Converting...", flush=True)
    model_moe.load_state_dict(model_moe_state_dict)
    model_moe = model_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"', flush=True)
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...", flush=True)
    config_moe.save_pretrained(save_path)
    model_moe.save_pretrained(save_path)
    print(f'Converted model saved to "{save_path}".', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--moe_config_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--split_file_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering")
    parser.add_argument('--save_path', type=str, default="/home/data/models/llama-moe-transformers/7B/")
    parser.add_argument('--template', type=str, default="layers.{}.mlp.up_proj.weight")

    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of selected experts')

    args = parser.parse_args()
    print(args, "\n")

    convert(
        args.model_path,
        args.moe_config_path,
        args.split_file_path,
        args.save_path,
        args.num_experts,
        args.num_selects,
        args.template
    )

    print("Done.")
