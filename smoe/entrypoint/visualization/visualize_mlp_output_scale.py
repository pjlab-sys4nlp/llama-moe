import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from smoe.data.collate_fn import tensor_dict_cat_collator
from smoe.data.datasets_moefication import LineByLineJsonlTextDataset
from smoe.models.llama_moe import LlamaMoEForCausalLM
from smoe.utils.model_operation.modify_llama_model import (
    llama_with_hidden_states_scale_recording,
)
from smoe.utils.model_operation.modify_llama_moe_model import (
    llama_moe_with_hidden_states_scale_recording,
)
from smoe.utils.operations.operation_string import str2bool

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data_begin_index', type=int, default=0)
    parser.add_argument('--data_end_index', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)  # 单次evaluate的batch_size
    parser.add_argument('--block_size', type=int, default=2048)  # 单次evaluate的seq_len
    parser.add_argument('--is_moe', type=str, default="False")
    parser.add_argument('--moe_score_scale_factor', type=float, default=None)  # 单次evaluate的batch_size

    args = parser.parse_args()
    args.is_moe = str2bool(args.is_moe)
    print("\n", args)

    print("\ncuda is_available: " + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    if args.is_moe:
        model = LlamaMoEForCausalLM.from_pretrained(args.model_path).model
        model = llama_moe_with_hidden_states_scale_recording(model)
        if args.moe_score_scale_factor is not None:
            model.set_moe_calculator_score_scale_factor(args.moe_score_scale_factor)
    else:
        model = LlamaForCausalLM.from_pretrained(args.model_path).model
        model = llama_with_hidden_states_scale_recording(model)

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

    """visualization & save"""
    avg_mlp_outputs = []
    avg_mlp_residuals = []

    for layer_idx, layer in enumerate(model.layers):
        avg_mlp_outputs.append(torch.mean(torch.cat(layer.mlp_outputs), dim=0).item())
        avg_mlp_residuals.append(torch.mean(torch.cat(layer.mlp_residuals), dim=0).item())

    print("avg_mlp_outputs:", avg_mlp_outputs, sep="\n")
    print("avg_mlp_residuals:", avg_mlp_residuals, sep="\n")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, "mlp_outputs.txt"), "w") as file:
        file.write(str(avg_mlp_outputs))
    with open(os.path.join(args.save_path, "mlp_residuals.txt"), "w") as file:
        file.write(str(avg_mlp_residuals))

    print("Done.")
