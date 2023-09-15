"""
Visualization of pair-wise overlap rate & overlap neuron count for moe models constructed by importance criterion (Share=True).
"""
import argparse
import os

import torch
from tqdm import tqdm

from smoe.utils.visualization.visualize import visualize_expert_neuron_overlap

# fmt: off
if __name__ == "__main__":
    torch.set_printoptions(
        precision=4,  # 精度，保留小数点后几位，默认4
        threshold=100000,
        edgeitems=3,
        linewidth=160,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile="full",
        sci_mode=False  # 用科学技术法显示数据，默认True
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--score_file_path', type=str, default="D:/dzdon/Desktop/llama_7B-Gradients-l1_norm-sample-feature_change")
    parser.add_argument('--save_path', type=str, default="D:/dzdon/Desktop/vis")
    parser.add_argument('--expert_size', type=int, default=688)
    parser.add_argument('--score_file_template', type=str, default="layers.{}.mlp.up_proj.weight.change")
    parser.add_argument('--criterion', type=str, default="max", choices=("min", "max"))

    args = parser.parse_args()
    print("\n", args)

    num_hidden_layers = 32
    intermediate_size = 11008

    for i in tqdm(range(num_hidden_layers)):
        """read scores from files"""
        score_list = []
        for expert_folder_name in os.listdir(args.score_file_path):
            grad_file = os.path.join(args.score_file_path, expert_folder_name, args.score_file_template.format(i))
            grad = torch.load(grad_file, map_location="cpu")
            score_list.append(grad)
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

        """overlap rate between each expert pair"""
        # rate calculation: intersection(Ei, Ej) / union(Ei, Ej)
        intersection_num = torch.mm(selected_masks, selected_masks.transpose(0, 1))
        union_num = torch.full_like(intersection_num, fill_value=intermediate_size) - torch.mm((1 - selected_masks), (1 - selected_masks).transpose(0, 1))
        overlap_rate = intersection_num / union_num

        # print(intersection_num)
        # print(union_num)
        print("overlap_rate", overlap_rate, sep="\n")

        """overlap count for each expert"""
        # rows: overlap count,  columns: different experts
        overlap_count = torch.zeros((num_experts, num_experts), dtype=torch.int)

        sum_count = selected_masks.sum(0)  # shape(intermediate_size,)
        selected_masks = selected_masks.bool()
        for overlap_times in range(num_experts):
            this_overlap_neurons = (sum_count == (overlap_times + 1))  # shape(intermediate_size,)
            # print(this_overlap_neurons.sum())
            each_expert_overlap_neurons = selected_masks & this_overlap_neurons  # shape(num_experts, intermediate_size)
            # print(each_expert_overlap_neurons.sum())
            overlap_count[overlap_times, :] = each_expert_overlap_neurons.sum(1)
            # print(overlap_count[overlap_times, :])

        print(overlap_count.sum(0))
        print("overlap_count", overlap_count, sep="\n")

        """save graphs"""
        total_neurons = (sum_count > 0).sum().item()
        visualize_expert_neuron_overlap(overlap_rate.numpy(), overlap_count.numpy(), total_neurons, args.expert_size, i, save_dir=args.save_path)

    print("done.")
