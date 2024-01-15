import argparse
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

from smoe.data.collate_fn import tensor_dict_cat_collator
from smoe.data.datasets_moe import LineByLineJsonlTextDataset
from smoe.models.llama_moe import LlamaMoEForCausalLM
from smoe.utils.model_operation.modify_llama_moe_model import (
    llama_moe_with_hidden_states_recording,
)
from smoe.utils.operations.operation_string import str2bool
from smoe.utils.seed import set_seed
from smoe.utils.visualization.visualize import visualize_expert_load_heatmap

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/llama_7B")
    parser.add_argument('--model_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_model_bak/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-23000/commoncrawl-part-000203-16de0c55-head1000.jsonl")
    parser.add_argument('--data_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/data/vis_data")
    parser.add_argument('--save_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/visualization")
    parser.add_argument('--save_name_prefix', type=str, default="")
    parser.add_argument('--reinit_gate', type=str, default="False")
    parser.add_argument('--data_begin_index', type=int, default=0)
    parser.add_argument('--data_end_index', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=8)  # 单次evaluate的batch_size
    parser.add_argument('--block_size', type=int, default=2048)  # 单次evaluate的seq_len
    parser.add_argument('--use_cpu', type=str, default="False")

    args = parser.parse_args()
    args.reinit_gate = str2bool(args.reinit_gate)
    args.use_cpu = str2bool(args.use_cpu)
    print("\n", args)

    print("\ncuda is_available: " + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"

    """load tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    """prepare datasets"""
    print("\nReading dataset from file \"" + args.data_path + "\"...")
    data_index_range = (args.data_begin_index, args.data_end_index)
    dataset = LineByLineJsonlTextDataset(tokenizer, file_path=args.data_path, block_size=args.block_size, data_index_range=data_index_range)
    print(f"Dataset: {sum([torch.sum(dataset[i]['attention_mask']).item() for i in range(len(dataset))])} total tokens.")  # 统计非special token的数量

    """prepare dataloader"""
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=tensor_dict_cat_collator, num_workers=8, pin_memory=True, persistent_workers=True)

    """load model"""
    print("Loading llama model...")
    model = LlamaMoEForCausalLM.from_pretrained(args.model_path).model
    model = llama_moe_with_hidden_states_recording(model, device)
    if args.reinit_gate:
        set_seed(0)
        model.reset_gate_network()

    """evaluation"""
    print("Start evaluation...")
    model.to(device)
    model.half()
    model.eval()
    iter_train = iter(data_loader)
    for step in tqdm(range(len(data_loader)), desc="forward step", position=0, leave=True):
        batch = next(iter_train)
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model(**batch)

    """visualization"""
    dataset_name = os.path.split(args.data_path)[1].split(".")[0] + args.save_name_prefix
    for layer_idx, layer in enumerate(model.layers):
        load_sum = layer.mlp.gate.load_sum.cpu().numpy()  # shape(num_experts)
        visualize_expert_load_heatmap(
            load_sum,
            layer_idx,
            dataset_name,
            save_dir=args.save_path + "-heat"
        )
        # visualize_expert_load_barv(
        #     load_sum,
        #     layer_idx,
        #     dataset_name,
        #     save_dir=args.save_path + "-bar"
        # )

    """save evaluation results as cache"""
    saved_dict = {
        "samples_cnt": model.layers[0].mlp.gate.samples_cnt,
        "importance_sum_list": [],
        "importance_loss_sum_list": [],
        "load_sum_list": [],
        "load_loss_sum_list": [],
    }
    importance_sum_list = []
    importance_loss_sum_list = []
    load_sum_list = []
    load_loss_sum_list = []

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        saved_dict["importance_sum_list"].append(layer.mlp.gate.importance_sum.cpu())
        saved_dict["importance_loss_sum_list"].append(layer.mlp.gate.importance_loss_sum.cpu())
        saved_dict["load_sum_list"].append(layer.mlp.gate.load_sum.cpu())
        saved_dict["load_loss_sum_list"].append(layer.mlp.gate.load_loss_sum.cpu())

    if not os.path.exists(args.save_path + "-results"):
        os.makedirs(args.save_path + "-results")
    torch.save(saved_dict, os.path.join(args.save_path + "-results", dataset_name + ".pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")
