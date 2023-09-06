from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def visualize_expert_load_heatmap(
    load_sum: np.ndarray,
    layer_idx: int,
    dataset_name: str,
    shape: tuple = (4, 4),
    save_dir: str = "results/expert_load_vis",
):
    save_dir_path = Path(save_dir)
    if save_dir_path.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    save_dir_path.mkdir(exist_ok=True, parents=True)
    path = save_dir_path / Path(f"{dataset_name}_Layer{layer_idx}.png")

    data = load_sum.reshape(*shape)

    cmap = mpl.colormaps["OrRd"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")

    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title(f"{dataset_name} - Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
    plt.close()


def visualize_expert_load_barv(
    load_sum: np.ndarray,
    layer_idx: int,
    dataset_name: str,
    y_max: float = None,
    x_label: str = None,
    save_dir: str = "results/expert_load_vis",
):
    save_dir_path = Path(save_dir)
    if save_dir_path.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    save_dir_path.mkdir(exist_ok=True, parents=True)
    path = save_dir_path / Path(f"{dataset_name}_Layer{layer_idx}.png")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = range(load_sum.shape[0])
    ax.bar(xs, load_sum)
    ax.set_xticks(xs)
    ax.set_title(f"{dataset_name} - Layer {layer_idx}")
    if y_max:
        ax.set_ylim([0, y_max])
    if x_label:
        ax.set_xlabel(x_label)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close()
