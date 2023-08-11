import argparse
import math
import os

import torch
from pebble import ProcessExpired, ProcessPool
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaConfig, LlamaModel

from smoe.data.collate_fn import separate_collater
from smoe.data.datasets_moefication import ShardDatasetForMoEGate
from smoe.utils.io import torch_load_template_file
from smoe.utils.moefication.expert_select import MLPGate
from smoe.utils.moefication.visualize import visualize_expert_select_mlp


# 多进程分词函数
def train_layer(args, train_layers, train_percent, batch_size, epochs, lr, device):
    # fmt: off
    print("Loading llama model...")
    model = LlamaModel.from_pretrained(args.model_path)

    for layer in train_layers:
        print(f"Training MoE Gate for layer {layer}...")

        """prepare datasets"""
        hidden_inputs_path = os.path.join(args.hidden_features_path, "hidden_inputs", "layer" + str(layer))
        if "gate_proj" in args.template:
            hidden_outputs_path = os.path.join(args.hidden_features_path, "hidden_gate_outputs", "layer" + str(layer))
        elif "up_proj" in args.template:
            hidden_outputs_path = os.path.join(args.hidden_features_path, "hidden_up_outputs", "layer" + str(layer))
        else:
            raise ValueError

        train_dataset = ShardDatasetForMoEGate(hidden_inputs_path, hidden_outputs_path,
                                               parallel_mode="workers", file_load_index_range=[0, int(train_percent * len(hidden_inputs_path))])
        valid_dataset = ShardDatasetForMoEGate(hidden_inputs_path, hidden_outputs_path,
                                               parallel_mode="workers", file_load_index_range=[int(train_percent * len(hidden_inputs_path)), len(hidden_inputs_path) - 1])

        """prepare dataloader"""
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=separate_collater, num_workers=16, pin_memory=True, persistent_workers=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=separate_collater, num_workers=8, pin_memory=True, persistent_workers=True)

        """prepare expert indices"""
        expert_indices = torch_load_template_file(args.split_file_path, args.template, layer)

        """train MLP"""
        if args.select_criterion == "l2_norm":
            criterion_config = {"threshold": 0.001}
        else:
            criterion_config = None

        center = MLPGate(args, model, train_loader, valid_loader, expert_indices, layer,
                         select_criterion=args.select_criterion, criterion_config=criterion_config)
        center.train(device, batch_size=batch_size, train_epochs=epochs, lr=lr, accumulate_steps=1,
                     use_balance=True, add_noise=False, use_softmax=args.use_softmax, balance_loss_lambda=0.0001)
    return train_layers
    # fmt: on


if __name__ == "__main__":
    # fmt: off
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("CUDA is_available: " + str(torch.cuda.is_available()), "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--split_file_path', type=str)
    parser.add_argument('--hidden_features_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_visualization_path', type=str, default="")
    parser.add_argument('--specify_layer', nargs='+', help='used to specify train layers, example \"--specify_layer 0 1 2 3\"')

    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--select_criterion', type=str, default='l2_norm', choices=["plain", "positive", "l2_norm"])
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of selected experts')
    parser.add_argument('--use_softmax', action='store_true')  # MLP Gate输出是否使用softmax激活

    parser.add_argument('--train_percent', type=float, default=0.95, help="percentage of training data")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Select-MLP-" + args.select_criterion)
    if args.save_visualization_path != "":
        args.save_visualization_path = os.path.join(args.save_visualization_path, os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Select-MLP-" + args.template.split(".")[3])
    if len(args.specify_layer) > torch.cuda.device_count():
        Warning(f"Number of GPUs({torch.cuda.device_count()}) is larger than number of layers({len(args.specify_layer)}), which will result in redundancy.")
    print(args, "\n")

    """load config"""
    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    """training configs"""
    devices = ["cuda:" + str(i) for i in range(torch.cuda.device_count())]
    devices = (devices * args.num_threads)[:args.num_threads]  # 平均分配gpu到各个进程
    print(devices)

    if "specify_layer" in args:  # 所有的层
        all_train_layers = [int(layer) for layer in args.specify_layer]
    else:
        all_train_layers = range(config.num_hidden_layers)

    layers_per_process = math.ceil(len(all_train_layers) / args.num_threads)  # 每个进程分到的层数，向上取整，防止有层分不到
    train_layers_list = []  # 每个进程分到的层
    for i in range(args.num_threads):
        train_layers_list.append(all_train_layers[:layers_per_process])
        all_train_layers = all_train_layers[layers_per_process:]
    print(train_layers_list)
    # fmt: on

    """train"""
    print("Training... (could be hours, please wait)")
    process_bar = tqdm(range(args.num_threads), desc="training process...")
    with ProcessPool(max_workers=args.num_threads) as pool:
        future = pool.map(
            train_layer,
            [args] * args.num_threads,
            train_layers_list,
            [args.train_percent] * args.num_threads,
            [args.batch_size] * args.num_threads,
            [args.epochs] * args.num_threads,
            [args.lr] * args.num_threads,
            devices,
        )
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                process_bar.update()
                print(f"Finished training for layer {result}!")
            except StopIteration:
                process_bar.update()
                break
            except TimeoutError as error:
                process_bar.update()
                print("TimeoutError:", error)
            except ProcessExpired as error:
                process_bar.update()
                print("ProcessExpired:", error)
            except Exception as error:
                process_bar.update()
                print("Exception:", error)
    process_bar.close()

    if args.save_visualization_path != "":
        visualize_expert_select_mlp(args.save_path, args.save_visualization_path)
    print("Done.")
