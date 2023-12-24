import argparse
import multiprocessing
import os
import pickle

import torch
from transformers import LlamaTokenizer

from smoe.data.datasets_moe import LineByLineJsonlTextDataset


# fmt: off
def process_dataset(args, tokenizer, key, file_name):
    raw_file_path = os.path.join(args.train_data_path, file_name)
    print("\nReading dataset \"" + key + "\" from raw file \"" + raw_file_path + "\"...")

    datasets = LineByLineJsonlTextDataset(tokenizer, file_path=raw_file_path, block_size=2048)

    if not os.path.exists(args.train_data_cache_path):
        os.makedirs(args.train_data_cache_path)

    cached_file_path = os.path.join(args.train_data_cache_path, key + "_cached.pth")
    torch.save(datasets.examples, cached_file_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dataset {key}: {sum([torch.sum(datasets[i]['attention_mask']).item() for i in range(len(datasets))])} total tokens.")  # 统计非special token的数量


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--train_data_path', type=str, default="/home/dongdz/workspace/moefication/llama_data/")
    parser.add_argument('--train_data_cache_path', type=str, default="/home/dongdz/workspace/moefication/llama_data_cache/")

    args = parser.parse_args()
    print(args, "\n")

    """load tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    """prepare datasets"""
    dataset_names = [
        "commoncrawl",
        "c4",
        "github",
        "wikipedia",
        "books",
        "arxiv",
        "stackexchange"
    ]

    # read datasets
    pool = multiprocessing.Pool(processes=len(dataset_names))
    for key in dataset_names:
        for file_name in os.listdir(args.train_data_path):
            if key in file_name and file_name.endswith(".jsonl"):
                pool.apply_async(process_dataset, args=(args, tokenizer, key, file_name))
    pool.close()
    pool.join()

    print("Done.")
