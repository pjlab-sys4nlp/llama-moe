import argparse
import os

from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.expert_construction.expert_split import RandomSplit

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--save_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/")
    parser.add_argument('--template', type=str, default='layers.{}.mlp.up_proj.weight')
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Split-Random")
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    for i in tqdm(range(config.num_hidden_layers)):
        split = RandomSplit(args, config, args.template, i)
        split.split()
        split.cnt()
        split.save()
    print("Done.")
