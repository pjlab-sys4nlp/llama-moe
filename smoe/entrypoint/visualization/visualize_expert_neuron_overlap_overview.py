"""
Visualization of global overlap rate & total overlap neurons under different "expert_size" for moe models constructed by importance criterion (Share=True).
"""
import argparse
import math
import os

import torch
from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.io import torch_load_template_score_file
from smoe.utils.visualization.line import line_plot_with_highlight

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--score_file_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--score_file_template', type=str, default="layers.{}.mlp.up_proj.weight.change")
    parser.add_argument('--criterion', type=str, default="max", choices=("min", "max"))

    args = parser.parse_args()
    print("\n", args)

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    """load score files to get num_experts"""
    score_list = torch_load_template_score_file(args.score_file_path, args.score_file_template, 0)
    num_experts = len(score_list)

    assert config.intermediate_size % num_experts == 0
    min_expert_size = config.intermediate_size // num_experts
    expert_size_list = list(range(min_expert_size, config.intermediate_size + 1, min_expert_size))
    print(len(expert_size_list))
    print(expert_size_list)

    independent_neuron_count = {}
    overlap_percent = {}
    overlap_repeated_rate = {}

    for layer_id in tqdm(range(config.num_hidden_layers)):
        # print(f"Layer {layer_id}:")

        """read scores from files"""
        score_list = torch_load_template_score_file(args.score_file_path, args.score_file_template, layer_id)
        scores = torch.stack(score_list, dim=0)
        scores_sum = scores.sum(0)

        """sort scores"""
        sort_index_list = []
        for score in score_list:
            if args.criterion == "min":
                sorted_score, index = torch.sort(score)
            elif args.criterion == "max":
                sorted_score, index = torch.sort(score, descending=True)
            else:
                raise NotImplementedError
            sort_index_list.append(index)

        # for i in range(len(sort_index_list)):
        #     print(f"Expert {i} index: {sort_index_list[i][:5]}")

        """initialize dict"""
        for expert_id in range(num_experts):
            if (expert_id + 1) not in overlap_percent:
                overlap_percent[expert_id + 1] = {}
            overlap_percent[expert_id + 1][f"layer{layer_id}"] = []
        overlap_repeated_rate[f"layer{layer_id}"] = []
        independent_neuron_count[f"layer{layer_id}"] = []

        """summarize results"""
        for expert_size in expert_size_list:

            """get selected mask"""
            selected_mask_list = []
            for index in sort_index_list:
                selected_mask = torch.zeros((config.intermediate_size,), dtype=torch.int)
                selected_mask[index[:expert_size]] += 1
                selected_mask_list.append(selected_mask)
            selected_masks = torch.stack(selected_mask_list, dim=0)  # shape(num_experts, intermediate_size)
            sum_count = selected_masks.sum(0)  # shape(intermediate_size,)

            """independent neuron count"""
            this_size_independent_neuron_count = (sum_count > 0).sum().item()
            independent_neuron_count[f"layer{layer_id}"].append(this_size_independent_neuron_count)

            """overlap percent"""
            for overlap_num in range(1, num_experts + 1):
                this_num_overlap_percent = (sum_count == overlap_num).sum().item() / this_size_independent_neuron_count
                overlap_percent[overlap_num][f"layer{layer_id}"].append(this_num_overlap_percent)

            """overlap repeated rate"""
            this_size_total_neuron_num = expert_size * num_experts
            this_size_total_overlap_repeated_neuron_num = 0
            for overlap_num in range(2, num_experts + 1):
                this_size_total_overlap_repeated_neuron_num += (overlap_num - 1) * (sum_count == overlap_num).sum().item()
            overlap_repeated_rate[f"layer{layer_id}"].append(this_size_total_overlap_repeated_neuron_num / this_size_total_neuron_num)

    """visualization"""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    line_plot_with_highlight(
        expert_size_list,
        independent_neuron_count,
        legend_columns=math.ceil(config.num_hidden_layers / 32),
        title="Number of Independent Neurons Selected from All Neurons in the Layer",
        xlabel="Expert Size",
        ylabel="Number of Neurons",
        save_path=os.path.join(args.save_path, "independent-neuron-count.png"),
    )

    for overlap_num in tqdm(range(1, num_experts + 1)):
        line_plot_with_highlight(
            expert_size_list,
            overlap_percent[overlap_num],
            legend_columns=math.ceil(config.num_hidden_layers / 32),
            title=f"Percentage of {overlap_num} Overlapped Neurons in All Selected Neurons",
            xlabel="Expert Size",
            ylabel=f"Percentage (num_overlap_{overlap_num}_neurons / num_total_selected_neurons)",
            save_path=os.path.join(args.save_path, f"{overlap_num}-overlap.png"),
        )

    line_plot_with_highlight(
        expert_size_list,
        overlap_repeated_rate,
        legend_columns=math.ceil(config.num_hidden_layers / 32),
        title="Percentage of Overlapped Neurons in the Neurons All Experts have",
        xlabel="Expert Size",
        ylabel="Ratio (overlap_repeated_neuron_num / total_neuron_num)",
        save_path=os.path.join(args.save_path, "overlap-rate.png"),
    )

    print("done.")
