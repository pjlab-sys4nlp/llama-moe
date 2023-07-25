import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

from smoe.data.collate_fn import separate_collater
from smoe.data.llama_moefication_datasets import ShardDatasetForMoEGate
from smoe.utils.io import torch_load_template_file
from smoe.utils.moefication.expert_select import MLPGate

# torch.multiprocessing.set_start_method('spawn')

print("CUDA is_available: " + str(torch.cuda.is_available()), "\n")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="/home/data/models/llama-transformers/7B"
)
parser.add_argument(
    "--split_file_path",
    type=str,
    default="/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering/",
)
parser.add_argument(
    "--hidden_features_path",
    type=str,
    default="/home/dongdz/workspace/moefication/llama_data/",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/home/dongdz/workspace/moefication/llama_moe_temp_files/",
)
parser.add_argument("--template", type=str, default="layers.{}.mlp.gate_proj.weight")
parser.add_argument(
    "--select_criterion",
    type=str,
    default="l2_norm",
    choices=["plain", "positive", "l2_norm"],
)
parser.add_argument("--num_experts", type=int, default=8, help="number of experts")
parser.add_argument("--num_selects", type=int, default=2, help="number of experts")
parser.add_argument(
    "--specify_layer",
    nargs="+",
    help='used to specify train layers, example "--specify_layer 0 1 2 3"',
)

args = parser.parse_args()
args.save_path = os.path.join(
    args.save_path,
    os.path.split(args.model_path)[1]
    + "-"
    + str(args.num_experts)
    + "Expert-Select-MLP-"
    + args.select_criterion,
)
print(args, "\n")

"""load model"""
print("Loading llama model...")
model = LlamaForCausalLM.from_pretrained(args.model_path).model

"""training"""
train_percent = 0.95
batch_size = 1024
epochs = 100
lr = 0.01
device = "cuda:0"

if "specify_layer" in args:
    train_layers = [int(layer) for layer in args.specify_layer]
else:
    train_layers = range(model.config.num_hidden_layers)

print(train_layers)
for layer in train_layers:
    print(f"Training MoE Gate for layer {layer}...")

    """prepare datasets"""
    hidden_inputs_path = os.path.join(
        args.hidden_features_path, "hidden_inputs", args.template.format(layer)
    )
    hidden_gate_outputs_path = os.path.join(
        args.hidden_features_path, "hidden_gate_outputs", args.template.format(layer)
    )

    train_dataset = ShardDatasetForMoEGate(
        hidden_inputs_path,
        hidden_gate_outputs_path,
        parallel_mode="workers",
        file_load_index_range=[0, int(train_percent * len(hidden_inputs_path))],
    )
    valid_dataset = ShardDatasetForMoEGate(
        hidden_inputs_path,
        hidden_gate_outputs_path,
        parallel_mode="workers",
        file_load_index_range=[
            int(train_percent * len(hidden_inputs_path)),
            len(hidden_inputs_path) - 1,
        ],
    )

    """prepare dataloader"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=separate_collater,
        num_workers=8,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=separate_collater,
        num_workers=8,
        pin_memory=True,
    )

    """prepare expert indices"""
    expert_indices = torch_load_template_file(
        args.split_file_path, args.template, layer
    )

    """train MLP"""
    if args.select_criterion == "l2_norm":
        criterion_config = {"threshold": 0.001}
    else:
        criterion_config = None

    center = MLPGate(
        args,
        model,
        train_loader,
        valid_loader,
        expert_indices,
        layer,
        select_criterion=args.select_criterion,
        criterion_config=criterion_config,
    )
    center.train(
        device, batch_size=batch_size, train_epochs=epochs, lr=lr, accumulate_steps=1
    )
print("Done.")
