import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

from smoe.data.collate_fn import tensor_dict_cat_collator
from smoe.data.datasets_moefication import LineByLineJsonlTextDataset
from smoe.models.llama_moe import LlamaMoEForCausalLM
from smoe.utils.model_operation.modify_llama_moe_model import llama_moe_with_hidden_states_scale_recording_early_stop
from smoe.utils.string_operation import str2bool

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--target_scale_file_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data_begin_index', type=int, default=0)
    parser.add_argument('--data_end_index', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)  # 单次evaluate的batch_size
    parser.add_argument('--block_size', type=int, default=2048)  # 单次evaluate的seq_len

    args = parser.parse_args()
    print("\n", args)

    print("\ncuda is_available: " + str(torch.cuda.is_available()), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """load tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    """prepare datasets"""
    print("\nReading dataset from file \"" + args.data_path + "\"...", flush=True)
    data_index_range = (args.data_begin_index, args.data_end_index)
    dataset = LineByLineJsonlTextDataset(tokenizer, file_path=args.data_path, block_size=args.block_size, data_index_range=data_index_range)
    print(f"Dataset: {sum([torch.sum(dataset[i]['attention_mask']).item() for i in range(len(dataset))])} total tokens.", flush=True)  # 统计非special token的数量

    """prepare dataloader"""
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=tensor_dict_cat_collator, num_workers=8, pin_memory=True, persistent_workers=True)

    """load target scale score files"""
    with open(os.path.join(args.target_scale_file_path, "mlp_outputs.txt"), "r") as file:
        mlp_outputs_list_str = file.readlines()[0]
        mlp_outputs_list = eval(mlp_outputs_list_str)

    with open(os.path.join(args.target_scale_file_path, "mlp_residuals.txt"), "r") as file:
        mlp_residuals_list_str = file.readlines()[0]
        mlp_residuals_list = eval(mlp_residuals_list_str)

    target_scale_gaps = [mlp_residuals_list[i] / mlp_outputs_list[i] for i in range(len(mlp_outputs_list))]

    """load model"""
    print("Loading llama model...", flush=True)
    model = LlamaMoEForCausalLM.from_pretrained(args.model_path).model
    model.set_moe_calculator_score_scale_factor(1.0)

    """calculate scale factor layer by layer"""
    print("Start evaluation...", flush=True)
    score_scale_factors = []

    model.to(device)
    model.half()
    model.eval()
    for layer_index in tqdm(range(model.config.num_hidden_layers), desc="forward by layer", leave=True):
        model = llama_moe_with_hidden_states_scale_recording_early_stop(model, early_stop_layer=layer_index)

        iter_train = iter(data_loader)
        for step in tqdm(range(len(data_loader)), desc="forward step", leave=False):
            batch = next(iter_train)
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                model(**batch)

        avg_mlp_output = torch.mean(torch.cat(model.layers[layer_index].mlp_outputs), dim=0).item()
        avg_mlp_residual = torch.mean(torch.cat(model.layers[layer_index].mlp_residuals), dim=0).item()
        this_layer_scale_gap = avg_mlp_residual / avg_mlp_output

        this_layer_score_scale_factor = this_layer_scale_gap / target_scale_gaps[layer_index]
        print(f"Layer {layer_index}: target_scale_gap={format(target_scale_gaps[layer_index], '.2f')}, layer_scale_gap={format(this_layer_scale_gap, '.2f')}, score_scale_factor={format(this_layer_score_scale_factor, '.2f')}, avg_mlp_output={format(avg_mlp_output, '.2f')}, avg_mlp_residual={format(avg_mlp_residual, '.2f')}\n", flush=True)
        # print(score_scale_factors, flush=True)
        model.layers[layer_index].mlp.calculator.score_scale_factor = this_layer_score_scale_factor
        score_scale_factors.append(this_layer_score_scale_factor)

    """save"""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, "score_scale_factors.txt"), "w") as file:
        file.write(str(score_scale_factors))

    print("Done.", flush=True)
