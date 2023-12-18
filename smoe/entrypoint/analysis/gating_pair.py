import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from smoe.data.streaming import CachedJsonlDataset
from smoe.models.llama_moe import LlamaMoEConfig, LlamaMoEForCausalLM
from smoe.utils.model_operation.modify_llama_moe_model import llama_moe_with_expert_load_pair_recording
from smoe.utils.operations.operation_tensor import move_tensors_to_device


def main(args):
    paths = Path(args.validation_dir).glob("*.jsonl")
    eval_dataset = {
        path.stem: CachedJsonlDataset(str(path), block_size=4096)
        for path in paths
    }
    print(f"eval types: {list(eval_dataset.keys())}")
    eval_dataset = ConcatDataset(eval_dataset.values())
    eval_dataset = [{key: torch.tensor(value) for key, value in data.items()} for data in eval_dataset]
    print(f"{len(eval_dataset)}")
    print(eval_dataset[0])

    model = LlamaMoEForCausalLM.from_pretrained(args.model_path)
    model.model = llama_moe_with_expert_load_pair_recording(model.model)

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=4)

    model.to("cuda")
    model.half()
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            sys.stderr.flush()
            move_tensors_to_device(batch, "cuda")
            model(**batch)
            if i >= 100:
                break

    all_records = {}
    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        all_records[layer_idx] = layer.mlp.gate.load_record

    torch.save(all_records, os.path.join(args.save_path, "records.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--validation_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main(args)
