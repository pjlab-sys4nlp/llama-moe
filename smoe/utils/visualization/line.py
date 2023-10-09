import matplotlib.pyplot as plt
import numpy as np

from smoe.utils.io import compress_png_image


def line_plot(
    xs,
    label_to_nums,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    save_path: str = None,
):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for label, nums in label_to_nums.items():
        ax.plot(xs, nums, label=label)
    ax.set_xticks(xs)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=320)
        compress_png_image(save_path, print_info=False)
    plt.close()


def line_plot_with_highlight(
    xs,
    label_to_nums,
    highlight_label_to_nums: dict = None,
    highlight_linewidth: int = 4,
    highlight_color: str = "black",
    cmap: str = "viridis",
    legend_columns: int = 1,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    save_path: str = None,
):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap(cmap)
    colors = np.linspace(0, 1, len(label_to_nums))

    for i, (label, nums) in enumerate(label_to_nums.items()):
        ax.plot(xs, nums, label=label, c=cmap(colors)[i, :3])

    if highlight_label_to_nums is not None:
        for i, (label, nums) in enumerate(highlight_label_to_nums.items()):
            ax.plot(
                xs, nums, label=label, linewidth=highlight_linewidth, c=highlight_color
            )

    ax.set_xticks(xs)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.legend(ncols=legend_columns)
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=320)
        compress_png_image(save_path, print_info=False)
    plt.close()
