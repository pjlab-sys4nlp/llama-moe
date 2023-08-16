import argparse
import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset

from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizer

from smoe.data.datasets_moefication import CommonDataset, LineByLineJsonlTextDataset
from smoe.utils.moefication.expert_split import GradientSplit

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--save_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/")
    parser.add_argument('--templates', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--datasets', nargs='+', help='datasets for gradient split, example \"--datasets commoncrawl c4 github\"')
    parser.add_argument('--data_use_range_begin', type=float, default=0)
    parser.add_argument('--data_use_range_end', type=float, default=1.0)

    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.save_path = os.path.join(args.save_path, os.path.split(args.model_path)[1] + "-" + str(args.num_experts) + "Expert-Split-Gradient")
    print(args, "\n")

    print("cuda is_available: " + str(torch.cuda.is_available()))
    dist.init_process_group(backend='nccl')

    """load tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    """prepare datasets"""
    # dataset_list = [
    #     "commoncrawl",
    #     "c4",
    #     "github",
    #     "wikipedia",
    #     "books",
    #     "arxiv",
    #     "stackexchange"
    # ]

    all_datasets = {}
    for key in tqdm(args.datasets):
        for file_name in os.listdir(args.train_data_path):
            if key in file_name and file_name.endswith(".jsonl"):
                cached_file_path = os.path.join(args.train_data_cache_path, key + "_cached.pth")
                if os.path.exists(cached_file_path):
                    print("\nReading dataset \"" + key + "\" from cached file \"" + cached_file_path + "\"...")
                    all_datasets[key] = CommonDataset(torch.load(cached_file_path))
                else:
                    raw_file_path = os.path.join(args.train_data_path, file_name)
                    print("\nReading dataset \"" + key + "\" from raw file \"" + raw_file_path + "\"...")
                    all_datasets[key] = LineByLineJsonlTextDataset(tokenizer, file_path=raw_file_path, block_size=2048)
                    if not os.path.exists(args.train_data_cache_path):
                        os.makedirs(args.train_data_cache_path)
                    with open(cached_file_path, "wb") as file:
                        torch.save(all_datasets[key].examples, os.path.join(args.train_data_cache_path, key + "_cached.pth"))
                print("Dataset " + key + ": " + str(sum([torch.sum(all_datasets[key][i] != 2).item() for i in range(len(all_datasets[key]))])) + " total tokens.")  # 统计非padding的token数量

    for key in args.datasets:  # reset the number of examples by data_use_range_begin and data_use_range_end
        random.seed(0)
        random.shuffle(all_datasets[key].examples)
        example_num = len(all_datasets[key].examples)
        all_datasets[key].examples = all_datasets[key].examples[int(example_num * args.data_use_range_begin):
                                                                int(example_num * args.data_use_range_end)]

    combined_dataset = []
    for key in args.datasets:
        combined_dataset.append(CommonDataset(all_datasets[key].examples))
        print("Dataset " + key + ": " + str(sum([torch.sum(combined_dataset[-1][i] != 2).item() for i in range(len(combined_dataset[-1]))])) + " used tokens.")  # 统计非padding的token数量
    combined_dataset = ConcatDataset(combined_dataset)

    """load model"""
    print("Loading llama model...")
    model = LlamaForCausalLM.from_pretrained(args.model_path).model

    """start splitting"""
    templates = args.templates.split(",")
    for template in templates:
        for i in tqdm(range(model.config.num_hidden_layers)):
            split = GradientSplit(args, model, template, i)
            split.split()
            split.cnt()
            split.save()
    print("Done.")
    # fmt: off
