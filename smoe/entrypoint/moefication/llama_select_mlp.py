import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

from smoe.data.collate_fn import tensor_list_cat_collator
from smoe.data.datasets_moefication import ShardDatasetForMoEGate
from smoe.utils.io import torch_load_template_file
from smoe.utils.moefication.expert_select import MLPGate
from smoe.utils.operations.operation_string import str2bool
from smoe.utils.visualization.visualize import visualize_expert_select_mlp

# fmt: off
if __name__ == "__main__":
    print("CUDA is_available: " + str(torch.cuda.is_available()), "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--split_file_path', type=str)
    parser.add_argument('--hidden_features_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_visualization_path', type=str, default="")
    parser.add_argument('--specify_layer', nargs='+', help='used to specify train layers, example \"--specify_layer 0 1 2 3\"')

    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--select_criterion', type=str, default='l2_norm', choices=["plain", "positive", "l1_norm", "l2_norm"])
    parser.add_argument('--mlp_init_criterion', type=str, default='weight', choices=["weight", "random"])
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of selected experts')

    parser.add_argument('--use_balance', type=str, default='False')
    parser.add_argument('--balance_loss_lambda', type=float, default=0.0)
    parser.add_argument('--add_noise', type=str, default='False')
    parser.add_argument('--use_softmax', type=str, default='False')  # MLP Gate输出是否使用softmax激活

    parser.add_argument('--data_use_percent', type=float, default=1.0, help="percentage of data file to use")
    parser.add_argument('--train_percent', type=float, default=0.95, help="percentage of training data")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)

    args = parser.parse_args()
    args.use_softmax = str2bool(args.use_softmax)
    args.save_path = os.path.join(args.save_path, os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Select-MLP-" + args.select_criterion+"-" + args.mlp_init_criterion)
    print(args, "\n")

    """load model"""
    print("Loading llama model...")
    model = LlamaForCausalLM.from_pretrained(args.model_path).model

    """training"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if "specify_layer" in args:
        train_layers = [int(layer) for layer in args.specify_layer]
    else:
        train_layers = range(model.config.num_hidden_layers)

    print(train_layers)
    for layer_idx in train_layers:
        print(f"Training MoE Gate for layer {layer_idx}...")

        """prepare datasets"""
        hidden_inputs_path = os.path.join(args.hidden_features_path, "hidden_inputs", "layer" + str(layer_idx))
        if "gate_proj" in args.template:
            hidden_outputs_path = os.path.join(args.hidden_features_path, "hidden_gate_outputs", "layer" + str(layer_idx))
        elif "up_proj" in args.template:
            hidden_outputs_path = os.path.join(args.hidden_features_path, "hidden_up_outputs", "layer" + str(layer_idx))
        else:
            raise ValueError

        train_dataset = ShardDatasetForMoEGate(hidden_inputs_path, hidden_outputs_path,
                                               parallel_mode="workers", data_use_percent=args.data_use_percent, file_load_index_range=[0, args.train_percent])
        valid_dataset = ShardDatasetForMoEGate(hidden_inputs_path, hidden_outputs_path,
                                               parallel_mode="workers", data_use_percent=args.data_use_percent, file_load_index_range=[args.train_percent, 1.0])

        """prepare dataloader"""
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=tensor_list_cat_collator, num_workers=16, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=tensor_list_cat_collator, num_workers=8, pin_memory=True)

        """prepare expert indices"""
        expert_indices = torch_load_template_file(args.split_file_path, args.template, layer_idx)

        """train MLP"""
        # if args.select_criterion == "l2_norm":
        #     criterion_config = {"threshold": 0.001}
        # else:
        #     criterion_config = None

        selector = MLPGate(args, model, train_loader, valid_loader, expert_indices, layer_idx,
                           select_criterion=args.select_criterion, mlp_init_criterion=args.mlp_init_criterion, criterion_config=None)
        selector.train(device, batch_size=args.batch_size, train_epochs=args.epochs, lr=args.lr, accumulate_steps=1,
                       use_balance=args.use_balance, add_noise=args.add_noise, use_softmax=args.use_softmax, balance_loss_lambda=args.balance_loss_lambda)

    if args.save_visualization_path != "":
        if "gate_proj" in args.template:
            proj_type = "gate_proj"
        elif "up_proj" in args.template:
            proj_type = "up_proj"
        else:
            raise ValueError
        visualize_expert_select_mlp(args.save_path, args.save_visualization_path, proj_type)
    print("Done.")
