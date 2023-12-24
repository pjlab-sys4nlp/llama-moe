import argparse
import os

import torch
from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.expert_construction.prune_llama import GradientPrune
from smoe.utils.io import torch_load_template_score_file
from smoe.utils.string_operation import str2bool

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--grad_file_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--expert_index', type=str)
    parser.add_argument('--retain_percent', type=float)
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    parser.add_argument('--kernel', type=str, default="plain", choices=("plain", "l1_norm", "l2_norm"))
    parser.add_argument('--accumulate_level', type=str, default="sample", choices=("sample", "total"))
    parser.add_argument('--importance_type', type=str, default="feature_grad", choices=("feature_grad", "feature_change"))
    parser.add_argument('--criterion', type=str, default="min", choices=("min", "max"))

    parser.add_argument('--use_grad_sum', type=str, default="False")

    args = parser.parse_args()
    args.use_grad_sum = str2bool(args.use_grad_sum)
    if args.expert_index != "All":
        args.expert_index = int(args.expert_index)
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)
    expert_size = int(config.intermediate_size * args.retain_percent)

    print("Processing layers...")
    save_root_path = args.save_path

    if args.importance_type == "feature_grad":
        file_postfix = ".grad"
    elif args.importance_type == "feature_change":
        file_postfix = ".change"
    else:
        raise NotImplementedError

    for i in tqdm(range(config.num_hidden_layers)):
        grad_list = torch_load_template_score_file(args.grad_file_path, args.template + file_postfix, i)

        if args.use_grad_sum:
            grad_list = torch.stack(grad_list, dim=0).sum(0)
        else:
            grad_list = grad_list[args.expert_index]

        args.save_path = os.path.join(
            save_root_path,
            f"{os.path.split(args.model_path)[1]}-Prune-Gradient-{args.criterion}-{args.kernel}-{args.accumulate_level}-{args.importance_type}",
            f"{args.expert_index}-{format(args.retain_percent, '.2f')}Percent-{expert_size}Neurons"
        )

        split = GradientPrune(args, args.template, i, grad_list)
        split.prune(expert_size, criterion=args.criterion)
        split.save()
    print("Done.")
    # fmt: on
