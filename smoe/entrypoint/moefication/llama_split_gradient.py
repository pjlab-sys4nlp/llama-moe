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
    parser.add_argument('--score_file_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--expert_num', type=int, default=None)
    parser.add_argument('--expert_size', type=int)
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    parser.add_argument('--kernel', type=str, default="plain", choices=("plain", "l1_norm", "l2_norm"))
    parser.add_argument('--accumulate_level', type=str, default="sample", choices=("sample", "total"))
    parser.add_argument('--criterion', type=str, default="min", choices=("min", "max"))
    parser.add_argument('--importance_type', type=str, default="feature_grad", choices=("feature_grad", "feature_change"))
    parser.add_argument('--share_neurons', type=str, default="False")

    args = parser.parse_args()
    args.share_neurons = str2bool(args.share_neurons)
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    print("Processing layers...")
    save_root_path = args.save_path

    if args.importance_type == "feature_grad":
        file_postfix = ".grad"
    elif args.importance_type == "feature_change":
        file_postfix = ".change"
    else:
        raise NotImplementedError

    for i in tqdm(range(config.num_hidden_layers)):
        score_list = []

        for expert_folder_name in os.listdir(args.score_file_path):
            score_file_path = os.path.join(args.score_file_path, expert_folder_name, args.template.format(i) + file_postfix)
            score = torch.load(score_file_path, map_location="cpu")
            score_list.append(score)
        # print(score_list)

        if args.expert_num is None:
            args.expert_num = len(score_list)
        else:
            assert args.expert_num <= len(score_list)

        args.save_path = os.path.join(
            save_root_path,
            f"{os.path.split(args.model_path)[1]}-Split-Gradient-{args.criterion}-{args.kernel}-{args.accumulate_level}-{args.importance_type}",
            f"{args.expert_num}Experts-{args.expert_size}Neurons{'-Share' if args.share_neurons else ''}"
        )

        split = GradientSplit(args, args.template, i, score_list)
        split.split(args.expert_num, args.expert_size, criterion=args.criterion, share_neurons=args.share_neurons)
        if not args.share_neurons:
            split.cnt()
        split.save()
    print("Done.")
    # fmt: on
