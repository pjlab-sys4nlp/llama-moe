import argparse
import os

from tqdm import tqdm
from transformers import LlamaForCausalLM

from smoe.utils.expert_construction.expert_split import GraphSplit

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model_path")
    parser.add_argument("--hidden_features_path", type=str, default="./hidden_features_path", )
    parser.add_argument("--save_path", type=str, default="./save_path", )
    parser.add_argument('--specify_layer', nargs='+', help='used to specify train layers, example \"--specify_layer 0 1 2 3\"')

    parser.add_argument("--template", type=str, default="layers.{}.mlp.gate_proj.weight")
    parser.add_argument("--num_experts", type=int, default=8, help="number of experts")
    parser.add_argument("--metric", type=str, default="l1_norm")
    parser.add_argument("--threshold", type=int, default=1)

    args = parser.parse_args()
    # args.save_path = os.path.join(
    #     args.save_path,
    #     os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Split-Graph-" + str(args.metric),
    # )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("Loading llama model...")
    model = LlamaForCausalLM.from_pretrained(args.model_path).model

    if "specify_layer" in args:
        train_layers = [int(layer) for layer in args.specify_layer]
    else:
        train_layers = range(model.config.num_hidden_layers)

    for layer_idx in train_layers:
        print(f"Creating co-activation matrix for layer {layer_idx}...")
        split = GraphSplit(args, model, args.template, layer_idx)
        split.split_and_save()
    print("Done.")
    # fmt: on
