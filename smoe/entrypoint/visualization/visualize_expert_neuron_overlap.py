"""
Visualization of pair-wise overlap rate & overlap neuron count for moe models constructed by importance criterion (Share=True).
"""
import argparse
import os

import torch
from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.io import delete_file_or_path, torch_load_template_score_file
from smoe.utils.visualization.visualize import visualize_expert_neuron_overlap

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--score_file_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--expert_size', type=int)
    parser.add_argument('--score_file_template', type=str, default="layers.{}.mlp.up_proj.weight.change")
    parser.add_argument('--criterion', type=str, default="max", choices=("min", "max"))

    args = parser.parse_args()
    print("\n", args)

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    delete_file_or_path(os.path.join(args.save_path, "total_neurons.txt"))

    for layer_id in tqdm(range(config.num_hidden_layers)):
        """read scores from files"""
        score_list = torch_load_template_score_file(args.score_file_path, args.score_file_template, layer_id)
        num_experts = len(score_list)
        scores = torch.stack(score_list, dim=0)

        """get selected mask"""
        selected_mask_list = []
        for j, score in enumerate(score_list):
            if args.criterion == "min":
                sorted_score, index = torch.sort(score)
            elif args.criterion == "max":
                sorted_score, index = torch.sort(score, descending=True)
            else:
                raise NotImplementedError
            selected_mask = torch.zeros_like(score, dtype=torch.int)
            selected_mask[index[:args.expert_size]] += 1
            selected_mask_list.append(selected_mask)
        selected_masks = torch.stack(selected_mask_list, dim=0)  # shape(num_experts, intermediate_size)

        """visualize"""
        visualize_expert_neuron_overlap(selected_masks, num_experts, config.intermediate_size, args.expert_size, layer_id, save_dir=args.save_path)

    print("done.")
