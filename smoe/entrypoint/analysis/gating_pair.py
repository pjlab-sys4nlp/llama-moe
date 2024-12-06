import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from smoe.data.streaming import CachedJsonlDataset
from smoe.models.llama_moe import LlamaMoEForCausalLM
from smoe.utils.io import compress_png_image
from smoe.utils.model_operation.modify_llama_moe_model import (
    llama_moe_with_expert_load_pair_recording,
)
from smoe.utils.operations.operation_tensor import move_tensors_to_device


def main(args):
    """data"""
    paths = Path(args.validation_dir).glob("*.jsonl")
    eval_dataset = {
        path.stem: CachedJsonlDataset(str(path), block_size=4096) for path in paths
    }
    print(f"eval types: {list(eval_dataset.keys())}")
    eval_dataset = ConcatDataset(eval_dataset.values())
    eval_dataset = [
        {key: torch.tensor(value) for key, value in data.items()}
        for data in eval_dataset
    ]
    print(f"{len(eval_dataset)}")
    print(eval_dataset[0])

    random.seed(0)
    random.shuffle(eval_dataset)

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=4)

    """model"""
    model = LlamaMoEForCausalLM.from_pretrained(args.model_path)
    model.model = llama_moe_with_expert_load_pair_recording(model.model)
    model.to("cuda")
    model.eval()

    """eval"""
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            sys.stderr.flush()
            move_tensors_to_device(batch, "cuda")
            model(**batch)
            if i >= 2000:
                break

    """aggregate"""
    all_records = {}
    for layer_idx, layer in enumerate(
        model.model.layers
    ):  # locate block by the name template
        all_records[layer_idx] = np.zeros(
            (model.config.num_experts, model.config.num_experts)
        )
        for pair in tqdm(
            layer.mlp.gate.load_record,
            desc=f"aggregating results for layer {layer_idx}",
        ):
            sys.stderr.flush()
            all_records[layer_idx][pair[0], pair[1]] += 1

    """save raw"""
    save_raw_pair_path = os.path.join(args.save_path, "raw_pair")
    if not os.path.exists(save_raw_pair_path):
        os.makedirs(save_raw_pair_path)

    for layer_idx, data in all_records.items():
        cmap = matplotlib.colormaps["OrRd"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")

        for i in range(model.config.num_experts):
            for j in range(model.config.num_experts):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=3,
                )

        ax.set_title(f"Layer {layer_idx}")
        ax.set_axis_off()
        fig.colorbar(im)
        fig.tight_layout()

        save_file = os.path.join(save_raw_pair_path, f"layer{layer_idx}.png")
        fig.savefig(save_file, dpi=320, bbox_inches="tight")
        compress_png_image(save_file, print_info=False)
        plt.close(fig)

    torch.save(all_records, os.path.join(save_raw_pair_path, "records.pt"))

    """save aggregated"""
    save_aggregate_pair_path = os.path.join(args.save_path, "aggregate_pair")
    if not os.path.exists(save_aggregate_pair_path):
        os.makedirs(save_aggregate_pair_path)

    for layer_idx in all_records.keys():
        ######################################
        all_records[layer_idx] += all_records[layer_idx].transpose()
        data = all_records[layer_idx]
        ######################################

        cmap = matplotlib.colormaps["OrRd"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")

        for i in range(model.config.num_experts):
            for j in range(model.config.num_experts):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

        ax.set_title(f"Layer {layer_idx}")
        ax.set_axis_off()
        fig.colorbar(im)
        fig.tight_layout()

        save_file = os.path.join(save_aggregate_pair_path, f"layer{layer_idx}.png")
        fig.savefig(save_file, dpi=320, bbox_inches="tight")
        compress_png_image(save_file, print_info=False)
        plt.close(fig)

    torch.save(all_records, os.path.join(save_aggregate_pair_path, "records.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--validation_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main(args)
