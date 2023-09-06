import argparse
import os

from tqdm import tqdm
from transformers import LlamaForCausalLM

from smoe.utils.moefication.expert_split import ClusteringSplit

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--save_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/")
    parser.add_argument('--templates', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--metric', type=str, default="l2", choices=("l2", "cos"))

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Split-Clustering-" + args.metric)

    print("Loading llama model...")
    model = LlamaForCausalLM.from_pretrained(args.model_path).model

    templates = args.templates.split(",")
    for template in templates:
        for i in tqdm(range(model.config.num_hidden_layers)):
            split = ClusteringSplit(args, model, template, i)
            split.split()
            split.cnt()
            split.save()
    print("Done.")
    # fmt: on
