import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def line_plot_with_highlight(
        xs,
        label_to_nums,
        highlight_label_to_nums=None,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        save_path: str = None,
):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap('viridis')
    colors = np.linspace(0, 1, len(label_to_nums))

    for i, (label, nums) in enumerate(label_to_nums.items()):
        ax.plot(xs, nums, label=label, c=cmap(colors)[i, :3])

    if highlight_label_to_nums is not None:
        for i, (label, nums) in enumerate(highlight_label_to_nums.items()):
            ax.plot(xs, nums, label=label, linewidth=4, c="black")

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
    plt.close()


if __name__ == "__main__":
    # fmt: off

    grad_file_path = "/mnt/petrelfs/share_data/quxiaoye/moefication_results/split/Gradients/llama_13B-Gradients-l1_norm-sample-feature_change"
    save_path = "/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization_change_13B/"
    file_postfix = ".change"  # grad change

    num_experts = 16
    num_layers = 40
    template = "layers.{}.mlp.up_proj.weight"

    # layers = range(0, num_layers, 4)
    # layers = range(0, num_layers, 2)
    layers = range(num_layers)

    select_num_list = [864 * (j + 1) for j in range(16)]
    overlap_percent = {}
    for i in range(num_experts):
        overlap_percent[i + 1] = {}
    avg_overlap_num = {}
    total_selected_count = {}

    for i in tqdm(layers):
        grad_list = []
        for expert_folder_name in os.listdir(grad_file_path):
            grad_file = os.path.join(grad_file_path, expert_folder_name, template.format(i) + file_postfix)
            grad = torch.load(grad_file, map_location="cpu")
            grad_list.append(grad)
        grad_sum = torch.stack(grad_list, dim=0).sum(0)

        grad_ascend_list = []
        grad_ascend_index_list = []
        grad_descend_list = []
        grad_descend_index_list = []

        for j, grad in enumerate(grad_list):
            grad_ascend, grad_ascend_index = torch.sort(grad)
            grad_descend, grad_descend_index = torch.sort(grad, descending=True)

            grad_ascend_list.append(grad_ascend)
            grad_ascend_index_list.append(grad_ascend_index)
            grad_descend_list.append(grad_descend)
            grad_descend_index_list.append(grad_descend_index)

        grad_sum_ascend, grad_sum_ascend_index = torch.sort(grad_sum)
        grad_sum_descend, grad_sum_descend_index = torch.sort(grad_sum, descending=True)

        # grad_ascend_list.append(grad_sum_ascend)
        # grad_ascend_index_list.append(grad_sum_ascend_index)
        # grad_descend_list.append(grad_sum_descend)
        # grad_descend_index_list.append(grad_sum_descend_index)

        print(f"Layer {i}:")

        for j in range(len(grad_list)):
            print(f"Expert {j} ascend: {grad_ascend_list[j][:5]}")
        print(f"Expert Total ascend: {grad_sum_ascend[:5]}")

        # for j in range(len(grad_list)):
        #     print(f"Expert {j} ascend_index: {grad_ascend_index_list[j][:10]}")
        # print(f"Expert Total ascend_index: {grad_sum_ascend_index[:10]}")
        #
        # for j in range(len(grad_list)):
        #     print(f"Expert {j} descend: {grad_descend_list[j][:5]}")
        # print(f"Expert Total descend: {grad_sum_descend[:5]}")
        #
        # for j in range(len(grad_list)):
        #     print(f"Expert {j} descend_index: {grad_descend_index_list[j][:10]}")
        # print(f"Expert Total descend_index: {grad_sum_descend_index[:10]}")

        for j in range(num_experts):
            overlap_percent[j + 1][f"layer{i}"] = []
            overlap_percent[j + 1][f"layer{i}"] = []
        avg_overlap_num[f"layer{i}"] = []
        total_selected_count[f"layer{i}"] = []

        for select_num in select_num_list:
            overlap_count = torch.zeros_like(grad_sum)
            for j, index in enumerate(grad_ascend_index_list):
                overlap_count[index[:select_num]] += 1

            this_total_count = (overlap_count > 0).sum().item()
            total_selected_count[f"layer{i}"].append(this_total_count)

            avg_overlap_num[f"layer{i}"].append(0)
            for j in range(num_experts):
                # print((overlap_count >= (j + 1)).sum().item(), total_selected_count)
                j_overlap_percent = (overlap_count >= (j + 1)).sum().item() / this_total_count
                overlap_percent[j + 1][f"layer{i}"].append(j_overlap_percent)

                if j >= 2:
                    avg_overlap_num[f"layer{i}"][-1] += j * (overlap_percent[j][f"layer{i}"][-1] - overlap_percent[j + 1][f"layer{i}"][-1]) * select_num
            avg_overlap_num[f"layer{i}"][-1] += num_experts * overlap_percent[num_experts][f"layer{i}"][-1] * select_num

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for j in tqdm(range(num_experts)):
        line_plot_with_highlight(
            select_num_list,
            overlap_percent[j + 1],
            title=f"{j + 1}-overlap",
            xlabel="expert size",
            ylabel="percent",
            save_path=os.path.join(save_path, f"{j + 1}-overlap.png"),
        )

    line_plot_with_highlight(
        select_num_list,
        avg_overlap_num,
        title="avg-overlap",
        xlabel="expert size",
        ylabel="number of neurons",
        save_path=os.path.join(save_path, "avg-overlap.png"),
    )

    line_plot_with_highlight(
        select_num_list,
        total_selected_count,
        title="total-count",
        xlabel="expert size",
        ylabel="number of neurons",
        save_path=os.path.join(save_path, "total-count.png"),
    )

print("Done.")
# fmt: on
