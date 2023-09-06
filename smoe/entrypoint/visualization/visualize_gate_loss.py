import re
import statistics as sts
from collections import defaultdict
from pathlib import Path

from smoe.utils.io import load_nums_from_txt
from smoe.utils.visualization.line import line_plot

if __name__ == "__main__":
    folders = [
        ["L2", "/mnt/petrelfs/zhutong/smoe/results/llama_7B_MoE_16Select4-l2_norm"],
        ["Random Params", "/mnt/petrelfs/zhutong/smoe/results/random_16select4_moe"],
        [
            "Random Split",
            "/mnt/petrelfs/zhutong/smoe/results/RandomSplit-l2_norm-llama_7B-16Select4-up_proj",
        ],
    ]
    output_fig_file = "results/gate_loss.png"

    label_to_nums = defaultdict(list)
    for name, folder in folders:
        folder_path = Path(folder)
        txt_files = list(folder_path.glob("gate_loss_R*_L*.txt"))
        regex = re.compile(r"gate_loss_R(\d+)_L(\d+).txt")
        layer_to_loss = defaultdict(list)
        for txt_file in txt_files:
            rank, layer = regex.search(str(txt_file)).groups()
            rank, layer = int(rank), int(layer)
            layer_to_loss[layer].extend(load_nums_from_txt(txt_file))

        layers = []
        for layer, losses in sorted(layer_to_loss.items(), key=lambda item: item[0]):
            layers.append(layer)
            label_to_nums[name].append(sts.mean(losses))

    line_plot(
        layers,
        label_to_nums,
        title="gate loss",
        xlabel="layer",
        ylabel="loss",
        save_path=output_fig_file,
    )
