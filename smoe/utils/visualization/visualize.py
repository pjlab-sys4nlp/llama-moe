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

from smoe.data.datasets_moefication import ShardDataset
from smoe.utils.kernel_function import pass_kernel_function
from smoe.utils.visualization.plotter import plotter


def visualize_expert_select_mlp(result_path, save_path, proj_type):
    # fmt: off
    """从acc与log文件中汇总结果，给出训练曲线"""
    layer_best_acc = {}
    layer_best_loss = {}
    layer_save_epoch = {}

    layer_train_epochs = {}
    layer_train_losses = {}
    layer_train_accs = {}
    layer_valid_epochs = {}
    layer_valid_losses = {}
    layer_valid_accs = {}

    for filename in tqdm(os.listdir(result_path), desc="loading files..."):
        if proj_type in filename:
            layer_index = -1  # 层号
            for split_part in filename.split("."):  # layers.0.mlp.up_proj.weight
                try:
                    layer_index = int(split_part)
                except:
                    pass

            if filename.endswith(".acc"):
                with open(os.path.join(result_path, filename), "r", encoding="UTF-8") as file:
                    lines = file.readlines()
                for line in lines:
                    if "best_acc" in line:
                        layer_best_acc[layer_index] = float(line.split(" ")[1])  # best_acc: 0.352923583984375
                    if "best_loss" in line:
                        layer_best_loss[layer_index] = float(line.split(" ")[1])  # best_loss: 0.0123456
                    if "save_epoch" in line:
                        layer_save_epoch[layer_index] = float(line.split(" ")[1])  # save_epoch: 79

            elif filename.endswith(".log"):
                with open(os.path.join(result_path, filename), "r", encoding="UTF-8") as file:
                    lines = file.readlines()
                if len(lines) > 6:  # auto-remove previous results
                    lines = lines[-6:]
                    with open(os.path.join(result_path, filename), "w", encoding="UTF-8") as file:
                        file.writelines(lines)
                for i in range(len(lines)):
                    line = lines[i]  # train_acc: ['0.2017', '0.1989', ..., '0.2409', '0.2328']
                    list_str = line.split(": ")[1].strip()  # ['0.2017', '0.1989', ..., '0.2409', '0.2328']
                    list_str = list_str[1:-1]  # '0.2017', '0.1989', ..., '0.2409', '0.2328'
                    list_elements = list_str.split(', ')  # '0.2017' '0.1989' ... '0.2409' '0.2328'
                    float_list = [float(element.replace("\'", "")) for element in list_elements]  # 0.2017 0.1989 ... 0.2409 0.2328
                    lines[i] = float_list
                layer_train_epochs[layer_index] = [int(element) for element in lines[0]]
                layer_train_losses[layer_index] = lines[1]
                layer_train_accs[layer_index] = lines[2]
                layer_valid_epochs[layer_index] = [int(element) for element in lines[3]]
                layer_valid_losses[layer_index] = lines[4]
                layer_valid_accs[layer_index] = lines[5]

            else:
                pass

    if len(layer_best_acc) == 0:
        print(f"No selection results in \"{result_path}\" for \"{proj_type}\"")
        return

    layer_num = max([key for key in layer_best_acc.keys()])
    epoch_num = max([max(layer_valid_epochs[key]) + 1 for key in layer_valid_epochs.keys()])
    train_step_per_epoch = len(layer_train_epochs[0]) // len(set(layer_train_epochs[0]))
    valid_step_per_epoch = len(layer_valid_epochs[0]) // len(set(layer_valid_epochs[0]))
    p = plotter()

    """收敛时valid 准确率——按层划分"""
    p.add_figure("best_acc", xlabel="layers", ylabel="acc", title="best acc (avg {:.4f})".format(np.mean(list(layer_best_acc.values())).item()))
    p.add_label("best_acc", "acc", dot_graph=False)
    for layer_index in range(layer_num):
        if layer_index in layer_best_acc.keys():
            p.add_data("best_acc", "acc", layer_index, layer_best_acc[layer_index])

    """收敛时valid loss——按层划分"""
    p.add_figure("best_loss", xlabel="layers", ylabel="loss", title="best loss (avg {:.4f})".format(np.mean(list(layer_best_loss.values())).item()))
    p.add_label("best_loss", "loss", dot_graph=False)
    for layer_index in range(layer_num):
        if layer_index in layer_best_loss.keys():
            p.add_data("best_loss", "loss", layer_index, layer_best_loss[layer_index])

    """收敛步数——按层划分"""
    p.add_figure("best_epoch", xlabel="layers", ylabel="epoch", title="best epoch (avg {:.1f})".format(np.mean(list(layer_save_epoch.values())).item()))
    p.add_label("best_epoch", "epoch", dot_graph=False)
    for layer_index in range(layer_num):
        if layer_index in layer_save_epoch.keys():
            p.add_data("best_epoch", "epoch", layer_index, layer_save_epoch[layer_index])

    """train准确率曲线——按epoch划分"""
    p.add_figure("train_acc", xlabel="epoch", ylabel="acc", title="train acc")
    for layer_index in layer_train_epochs.keys():
        p.add_label("train_acc", "layer_{}".format(layer_index), markersize=4, dot_graph=False)
        for epoch in range(epoch_num):
            p.add_data("train_acc", "layer_{}".format(layer_index),
                       epoch, np.mean(layer_train_accs[layer_index][epoch * train_step_per_epoch:(epoch + 1) * train_step_per_epoch]).item())

    """valid准确率曲线——按epoch划分"""
    p.add_figure("valid_acc", xlabel="epoch", ylabel="acc", title="valid acc")
    for layer_index in layer_valid_epochs.keys():
        p.add_label("valid_acc", "layer_{}".format(layer_index), markersize=4, dot_graph=False)
        for epoch in range(epoch_num):
            p.add_data("valid_acc", "layer_{}".format(layer_index),
                       epoch, np.mean(layer_valid_accs[layer_index][epoch * valid_step_per_epoch:(epoch + 1) * valid_step_per_epoch]).item())

    """train loss曲线——按epoch划分"""
    p.add_figure("train_loss", xlabel="epoch", ylabel="loss", title="train loss")
    for layer_index in layer_train_epochs.keys():
        p.add_label("train_loss", "layer_{}".format(layer_index), markersize=4, dot_graph=False)
        for epoch in range(epoch_num):
            p.add_data("train_loss", "layer_{}".format(layer_index),
                       epoch, np.mean(layer_train_losses[layer_index][epoch * train_step_per_epoch:(epoch + 1) * train_step_per_epoch]).item())

    """valid loss曲线——按epoch划分"""
    p.add_figure("valid_loss", xlabel="epoch", ylabel="loss", title="valid loss")
    for layer_index in layer_valid_epochs.keys():
        p.add_label("valid_loss", "layer_{}".format(layer_index), markersize=4, dot_graph=False)
        for epoch in range(epoch_num):
            p.add_data("valid_loss", "layer_{}".format(layer_index),
                       epoch, np.mean(layer_valid_losses[layer_index][epoch * valid_step_per_epoch:(epoch + 1) * valid_step_per_epoch]).item())

    p.save(path=save_path, close_graph=True, bbox_inches="tight")
    print(f"Results saved to \"{save_path}\"!")
    # fmt: on


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
    # neuron_type 与 layer_idx 仅为生成图像名称使用

    # 划分 bin
    bin_edges = torch.linspace(edge[0], edge[1], num_bins + 1, device="cpu")  # 自定义 bin 的范围和数量

    # 准备数据集
    dataset = ShardDataset(hidden_outputs_path, parallel_mode="workers")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    iterator = iter(dataloader)

    # 读取数据
    total_bin_counts = np.zeros(num_bins)
    for step in tqdm(range(len(dataloader)), desc="iterating over data", leave=False):
        if step >= len(dataloader):
            break
        hidden_outputs = next(iterator).float().squeeze(0).to(device)
        hidden_outputs = pass_kernel_function(hidden_outputs, criterion=criterion)  # 按照指标转化
        bin_counts = torch.histc(hidden_outputs, bins=num_bins, min=edge[0], max=edge[1])  # 使用 torch.histc 进行 bin 统计
        total_bin_counts += bin_counts.cpu().numpy()

    # 使用Matplotlib绘制柱状图
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
    print(f'Results saved to "{save_path}"!')
    # fmt: on


def find_factors_with_minimal_sum(number):
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
    # Save the plot to a PNG in memory.
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
        im = ax.imshow(
            data[i].cpu().reshape(*shape).float().numpy(),
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
                    f"{data[i, row, col]:.5f}",
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
    path = save_dir_path / Path(f"{dataset_name}_Layer{layer_idx}.png")

    data = load_sum.reshape(*shape)

    cmap = mpl.colormaps["OrRd"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")

    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(j, i, f"{data[i, j]:.5f}", ha="center", va="center", color="black")

    ax.set_title(f"{dataset_name} - Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()
    if save_fig:
        fig.savefig(path, dpi=320, bbox_inches="tight")
    return fig


def visualize_expert_neuron_overlap(
    overlap_rate: np.ndarray,
    overlap_count: np.ndarray,
    total_neurons: int,
    expert_size: int,
    layer_idx: int,
    save_dir: str = "./",
    save_fig: bool = True,
):
    path_overlap_rate = Path(os.path.join(save_dir, "overlap_rate"))
    if path_overlap_rate.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    path_overlap_rate.mkdir(exist_ok=True, parents=True)

    path_overlap_count = Path(os.path.join(save_dir, "overlap_count"))
    if path_overlap_count.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    path_overlap_count.mkdir(exist_ok=True, parents=True)

    """overlap_rate"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        overlap_rate,
        vmin=0.0,
        vmax=1.0,
        cmap=mpl.colormaps["Greens"],
        interpolation="nearest",
    )

    for i in range(overlap_rate.shape[0]):
        for j in range(overlap_rate.shape[1]):
            ax.text(
                j,
                i,
                f"{overlap_rate[i, j]:.4f}",
                ha="center",
                va="center",
                color="black",
                fontsize=4,
            )

    ax.set_title(f"Total Selected Neurons {total_neurons} -- Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()

    if save_fig:
        fig.savefig(
            path_overlap_rate / Path(f"overlap_rate_layer{layer_idx}.png"),
            dpi=480,
            bbox_inches="tight",
        )
    plt.close(fig)

    """overlap_count"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        overlap_count,
        vmin=0,
        vmax=expert_size,
        cmap=mpl.colormaps["Blues"],
        interpolation="nearest",
    )

    for i in range(overlap_count.shape[0]):
        for j in range(overlap_count.shape[1]):
            ax.text(
                j,
                i,
                f"{overlap_count[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=4,
            )

    ax.set_title(f"Expert Size {expert_size} -- Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()

    if save_fig:
        fig.savefig(
            path_overlap_count / Path(f"overlap_count_layer{layer_idx}.png"),
            dpi=480,
            bbox_inches="tight",
        )
    plt.close(fig)


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
