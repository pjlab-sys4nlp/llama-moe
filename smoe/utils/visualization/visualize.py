import io
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from smoe.data.datasets_moe import ShardDataset
from smoe.utils.io import compress_png_image
from smoe.utils.operations.operation_tensor import pass_kernel_function
from smoe.utils.visualization.plotter import plotter


def visualize_swiglu_output(
    hidden_outputs_path,
    save_path,
    neuron_type,
    layer_idx,
    criterion="plain",
    num_bins=1000,
    edge=(-1.0, 1.0),
    device="cpu",
):
    # fmt: off
    # neuron_type and layer_idx are only used for generating the image filename

    # Define bins
    bin_edges = torch.linspace(edge[0], edge[1], num_bins + 1, device="cpu")  # Custom range and number of bins

    # Prepare dataset
    dataset = ShardDataset(hidden_outputs_path, parallel_mode="workers")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    iterator = iter(dataloader)

    # Read data
    total_bin_counts = np.zeros(num_bins)
    for step in tqdm(range(len(dataloader)), desc="iterating over data", leave=False):
        if step >= len(dataloader):
            break
        hidden_outputs = next(iterator).float().squeeze(0).to(device)
        hidden_outputs = pass_kernel_function(hidden_outputs, criterion=criterion)  # Transform according to the criterion
        bin_counts = torch.histc(hidden_outputs, bins=num_bins, min=edge[0], max=edge[1])  # Use torch.histc for bin counting
        total_bin_counts += bin_counts.cpu().numpy()

    # Plot histogram using Matplotlib
    fig_name = f"layer{layer_idx}_{neuron_type}_{criterion}"
    fig = plt.figure(fig_name)
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(bin_edges[:-1], total_bin_counts, width=(bin_edges[1] - bin_edges[0]), align="edge", alpha=0.7)
    ax.set_xlabel("SiwGLU Output")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of SiwGLU Output ({neuron_type}) ({criterion})")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, fig_name + ".png"), dpi=640, bbox_inches="tight")
    plt.close(fig)
    compress_png_image(os.path.join(save_path, fig_name + ".png"), print_info=False)
    print(f'Results saved to "{save_path}"!')
    # fmt: on


def find_factors_with_minimal_sum(number):
    if number == 1:
        return (1, 1)

    # Initialize variables to keep track of the factors with the minimal sum
    min_sum = float("inf")
    min_factors = None

    # Iterate through potential factors from 1 to half of the number
    for factor1 in range(1, number // 2 + 1):
        factor2 = number // factor1

        # Check if factor1 * factor2 is equal to the original number
        if factor1 * factor2 == number:
            current_sum = factor1 + factor2

            # Update the minimum sum and factors if the current sum is smaller
            if current_sum < min_sum:
                min_sum = current_sum
                min_factors = (factor1, factor2)

    return min_factors


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a image in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to PyTorch tensor
    image = Image.open(buf)
    # Convert PIL Image to a NumPy array
    numpy_array = np.array(image)
    # Convert the NumPy array to a PyTorch tensor
    torch_image = torch.from_numpy(numpy_array)
    # # Ensure the data type and normalize if needed (optional)
    # torch_image = torch_image.float() / 255.0  # Normalize to [0, 1] if the image has pixel values in [0, 255]
    # Convert to CHW format by rearranging the dimensions
    torch_image_chw = torch_image.permute(2, 0, 1)

    return torch_image_chw


def vis_tuple_heatmaps(tensors: tuple[torch.FloatTensor]):
    if (
        len(tensors) == 0
        or not all(isinstance(t, torch.Tensor) for t in tensors)
        or not all(t.shape == tensors[0].shape for t in tensors)
    ):
        return None
    data = torch.stack(tensors, dim=0)
    shape = find_factors_with_minimal_sum(data[0].numel())
    data = data.reshape(-1, *shape)
    img_grid = find_factors_with_minimal_sum(data.shape[0])

    cmap = mpl.colormaps["OrRd"]
    fig, axes = plt.subplots(*img_grid, figsize=[el * 5 for el in img_grid[::-1]])
    axes = axes.reshape(*img_grid)
    for i in range(data.shape[0]):
        ax = axes[i // img_grid[1], i % img_grid[1]]
        ax.imshow(
            data[i].cpu().reshape(*shape).float().detach().numpy(),
            cmap=cmap,
            interpolation="nearest",
            # vmin=0.0,
            # vmax=1.0,
        )
        for row in range(shape[0]):
            for col in range(shape[1]):
                ax.text(
                    col,
                    row,
                    f"{data[i, row, col]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_title(f"Layer {i}")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def get_heatmap_img_grid_for_tb(tensors: tuple[torch.FloatTensor]):
    fig = vis_tuple_heatmaps(tensors)
    if fig is None:
        return None
    img = plot_to_image(fig)
    return img


def visualize_expert_load_heatmap(
    load_sum: np.ndarray,
    layer_idx: int,
    dataset_name: str,
    shape: tuple = (4, 4),
    save_dir: str = "results/expert_load_vis",
    save_fig: bool = True,
):
    save_dir_path = Path(os.path.join(save_dir, f"layer{layer_idx}"))
    if save_dir_path.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    save_dir_path.mkdir(exist_ok=True, parents=True)
    # path = save_dir_path / Path(f"{dataset_name}_Layer{layer_idx}.pdf")
    # print(layer_idx, path)
    path = save_dir_path / Path(f"{dataset_name}_Layer{layer_idx}.png")

    data = load_sum.reshape(*shape)

    cmap = mpl.colormaps["OrRd"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    # im = ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=3500, vmax=4500)

    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", color="black")

    ax.set_title(f"{dataset_name} - Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()
    if save_fig:
        fig.savefig(str(path), dpi=320, bbox_inches="tight")
        if path.suffix == ".png":
            compress_png_image(str(path), print_info=False)
    return fig


def visualize_expert_neuron_overlap(
    selected_masks: torch.Tensor,
    num_experts: int,
    intermediate_size: int,
    expert_size: int,
    layer_idx: int,
    save_dir: str = "./",
    save_fig: bool = True,
):
    # fmt: off
    torch.set_printoptions(
        precision=4,  # 精度，保留小数点后几位，默认4
        threshold=100000,
        edgeitems=3,
        linewidth=160,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile="full",
        sci_mode=False  # 用科学技术法显示数据，默认True
    )

    """overlap rate between each expert pair"""
    # rate calculation: intersection(Ei, Ej) / union(Ei, Ej)
    intersection_num = torch.mm(selected_masks, selected_masks.transpose(0, 1))
    union_num = torch.full_like(intersection_num, fill_value=intermediate_size) - torch.mm((1 - selected_masks), (1 - selected_masks).transpose(0, 1))
    overlap_rate = intersection_num / union_num

    # print(intersection_num)
    # print(union_num)
    print("overlap_rate", overlap_rate, sep="\n", flush=True)

    """overlap count for each expert"""
    # rows: overlap count,  columns: different experts
    overlap_count = torch.zeros((num_experts, num_experts), dtype=torch.int)

    sum_count = selected_masks.sum(0)  # shape(intermediate_size,)
    selected_masks = selected_masks.bool()
    for overlap_times in range(num_experts):
        this_overlap_neurons = (sum_count == (overlap_times + 1))  # shape(intermediate_size,)
        # print(this_overlap_neurons.sum())
        each_expert_overlap_neurons = selected_masks & this_overlap_neurons  # shape(num_experts, intermediate_size)
        # print(each_expert_overlap_neurons.sum())
        overlap_count[overlap_times, :] = each_expert_overlap_neurons.sum(1)
        # print(overlap_count[overlap_times, :])

    # print(overlap_count.sum(0))
    print("overlap_count", overlap_count, sep="\n", flush=True)

    """save graphs"""
    total_neurons = (sum_count > 0).sum().item()
    overlap_rate = overlap_rate.numpy()
    overlap_count = overlap_count.numpy()

    path_overlap_rate = Path(os.path.join(save_dir, "overlap_rate"))
    if path_overlap_rate.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    path_overlap_rate.mkdir(exist_ok=True, parents=True)

    path_overlap_count = Path(os.path.join(save_dir, "overlap_count"))
    if path_overlap_count.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    path_overlap_count.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(save_dir, "total_neurons.txt"), "a") as file:
        file.write(f"{total_neurons}\n")

    """overlap_rate"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(overlap_rate, vmin=0.0, vmax=1.0, cmap=mpl.colormaps["Greens"], interpolation="nearest", )

    for i in range(overlap_rate.shape[0]):
        for j in range(overlap_rate.shape[1]):
            ax.text(j, i, f"{overlap_rate[i, j]:.4f}", ha="center", va="center", color="black", fontsize=4, )

    ax.set_title(f"Total Selected Neurons {total_neurons} -- Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()

    if save_fig:
        fig.savefig(path_overlap_rate / Path(f"overlap_rate_layer{layer_idx}.png"), dpi=480, bbox_inches="tight", )
        compress_png_image(str(path_overlap_rate / Path(f"overlap_rate_layer{layer_idx}.png")), print_info=False)
    plt.close(fig)

    """overlap_count"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(overlap_count, vmin=0, vmax=expert_size, cmap=mpl.colormaps["Blues"], interpolation="nearest", )

    for i in range(overlap_count.shape[0]):
        for j in range(overlap_count.shape[1]):
            ax.text(j, i, f"{overlap_count[i, j]}", ha="center", va="center", color="black", fontsize=4, )

    ax.set_title(f"Expert Size {expert_size} -- Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()

    if save_fig:
        fig.savefig(path_overlap_count / Path(f"overlap_count_layer{layer_idx}.png"), dpi=480, bbox_inches="tight", )
        compress_png_image(str(path_overlap_count / Path(f"overlap_count_layer{layer_idx}.png")), print_info=False)
    plt.close(fig)
    # fmt: on


def visualize_expert_load_barv(
    load_sum: np.ndarray,
    layer_idx: int,
    dataset_name: str,
    y_max: float = None,
    x_label: str = None,
    save_dir: str = "results/expert_load_vis",
):
    save_dir_path = Path(os.path.join(save_dir, f"layer{layer_idx}"))
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
    fig.savefig(path, dpi=320, bbox_inches="tight")
    compress_png_image(path, print_info=False)
