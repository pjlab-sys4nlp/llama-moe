import argparse
import os

import torch.cuda
from transformers import LlamaConfig

from smoe.utils.moefication.visualize import visualize_swiglu_output

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--hidden_features_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--specify_layer', nargs='+', help='used to specify layers for visualization, example \"--specify_layer 0 1 2 3\"')
    parser.add_argument('--visualize_criterion', default='plain', choices=["plain", "l1_norm", "l2_norm"])

    args = parser.parse_args()
    print(args, "\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    if "specify_layer" in args:
        visualize_layers = [int(layer) for layer in args.specify_layer]
    else:
        visualize_layers = range(config.num_hidden_layers)
    print(visualize_layers)

    for layer_idx in visualize_layers:
        print(f"Visualizing SiwGLU output for layer {layer_idx}...")

        if "gate_proj" in args.template:
            hidden_outputs_path = os.path.join(args.hidden_features_path, "hidden_gate_outputs", "layer" + str(layer_idx))
            neuron_type = "gate_proj"
        elif "up_proj" in args.template:
            hidden_outputs_path = os.path.join(args.hidden_features_path, "hidden_up_outputs", "layer" + str(layer_idx))
            neuron_type = "up_proj"
        else:
            raise ValueError

        if args.visualize_criterion == "plain":
            edge = (-0.5, 0.5)
        elif args.visualize_criterion == "l1_norm":
            edge = (0, 0.5)
        elif args.visualize_criterion == "l2_norm":
            edge = (0, 0.25)
        else:
            raise ValueError

        visualize_swiglu_output(hidden_outputs_path, args.save_path, neuron_type, layer_idx, criterion=args.visualize_criterion,
                                num_bins=1000, edge=edge, device=device)
