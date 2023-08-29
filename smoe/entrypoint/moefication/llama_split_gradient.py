import argparse
import os

import torch
from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.moefication.expert_split import GradientSplit
from smoe.utils.string_operation import str2bool

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--grad_file_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--expert_size', type=int)
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    parser.add_argument('--kernel', type=str, default="plain", choices=("plain", "l1_norm", "l2_norm"))
    parser.add_argument('--accumulate_level', type=str, default="sample", choices=("sample", "total"))
    parser.add_argument('--criterion', type=str, default="min", choices=("min", "max"))
    parser.add_argument('--share_neurons', type=str, default="False")

    args = parser.parse_args()
    args.share_neurons = str2bool(args.share_neurons)
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    print("Processing layers...")
    for i in tqdm(range(config.num_hidden_layers)):
        grad_list = []

        for expert_folder_name in os.listdir(args.grad_file_path):
            grad_file_path = os.path.join(args.grad_file_path, expert_folder_name, args.template.format(i) + ".grad")
            grad = torch.load(grad_file_path, map_location="cpu")
            grad_list.append(grad)
        print(grad_list)

        expert_num = len(grad_list)

        save_path = os.path.join(
            args.save_path,
            f"{os.path.split(args.model_path)[1]}-Split-Gradient-{args.criterion}-{args.kernel}-{args.accumulate_level}",
            f"{expert_num}Experts-{args.expert_size}Neurons-{'Share' if args.share_neurons else ''}"
        )

        split = GradientSplit(args, args.template, i, grad_list)
        split.split(args.expert_size, criterion=args.criterion, share_neurons=args.share_neurons)
        split.cnt()
        split.save()
    print("Done.")
    # fmt: on
