import argparse
import os

import tqdm
from transformers import LlamaForCausalLM

from smoe.utils.moefication.expert_split import GraphSplit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model_path")
    parser.add_argument("--metric", type=str, default="l1_norm")
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument(
        "--hidden_features_path",
        type=str,
        default="./hidden_features_path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./save_path",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="layers.{}.mlp.gate_proj.weight",
        help=(
            "weight names of the first linear layer in each FFN (use comma to separate"
            " multiple templates)"
        ),
    )
    parser.add_argument("--num_experts", type=int, default=8, help="number of experts")

    args = parser.parse_args()
    # args.save_path = os.path.join(
    #     args.save_path,
    #     os.path.split(args.model_path)[1]
    #     + "-"
    #     + str(args.num_experts)
    #     + "Expert-Split-Graph"
    #     + '-'
    #     + str(args.metric),
    # )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("Loading llama model...")
    model = LlamaForCausalLM.from_pretrained(args.model_path).model

    templates = args.templates.split(",")
    for template in templates:
        for i in tqdm.tqdm(range(model.config.num_hidden_layers)):
            print("now is layer: ", str(i))
            split = GraphSplit(args, model, template, i)
            split.split_and_save()
    print("Done.")
