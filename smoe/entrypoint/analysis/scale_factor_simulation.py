import statistics as sts
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def kaiming_init(*size):
    tensor = torch.randn(*size)
    nn.init.kaiming_normal_(tensor, mode="fan_out")
    return tensor


def simulation():
    xs = []
    ys = []

    init_func = torch.randn
    # init_func = kaiming_init

    max_num_experts = 32
    intermediate_size = 11008
    hidden_size = 4096
    base = None
    for k in range(1, max_num_experts + 1):
        mid = int(intermediate_size / k)
        distances = []
        for _ in range(10):
            gate = init_func(hidden_size, mid)
            up = init_func(hidden_size, mid)
            down = init_func(mid, hidden_size)

            x = init_func(1, hidden_size)
            # y = x @ l1 @ l2
            y = (F.silu(x @ gate) * (x @ up)) @ down

            dist = (x - y).abs().sum()
            # dist = (x - y).pow(2).sum()
            distances.append(dist.item())

        xs.append(k)
        if base is None and k == 1:
            base = sts.mean(distances)
        ys.append(base / sts.mean(distances))
        print(xs[-1], ys[-1])

    plt.plot(xs, ys, label="simulation")
    plt.plot(xs, np.sqrt(xs), label="sqrt", linestyle="dashed")
    plt.legend()
    plt.xlabel("#Experts")
    plt.ylabel("Scale Factor")
    plt.grid(True, zorder=-1)

    # plt.title("SwiGLU Kaiming Normal Initialization (fan_out)")
    # plt.savefig("swiglu_kaiming_fan_out_1024.png")

    out_dir = Path("results/analysis_scale_factor")
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.title("Normal Initialization")
    plt.savefig(out_dir / "normal.png")
    # plt.savefig(out_dir / "normal_dropout_rescale.png")


def line_graph(vals: list, label=None):
    plt.hist(vals, bins=100, density=True)
    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt)
    lnspc = np.linspace(xmin, xmax, len(vals))
    ab, bb, cb, db = stats.beta.fit(vals)
    pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
    plt.plot(lnspc, pdf_beta, label=label)
    return lnspc, pdf_beta


def stat_plot():
    dense_hidden = (
        torch.load("results/analysis_scale_factor/hidden_states.pt")
        .detach()
        .numpy()
        .flatten()
    )
    dense_hidden_x, dense_hidden_y = line_graph(dense_hidden, label="dense_hidden")
    dense_residual = (
        torch.load("results/analysis_scale_factor/residual.pt")
        .detach()
        .numpy()
        .flatten()
    )
    dense_residual_x, dense_residual_y = line_graph(
        dense_residual, label="dense_residual"
    )
    moe_hidden = (
        torch.load("results/analysis_scale_factor/moe_hidden_states.pt")
        .detach()
        .numpy()
        .flatten()
    )
    moe_hidden_x, moe_hidden_y = line_graph(moe_hidden, label="moe_hidden")
    moe_residual = (
        torch.load("results/analysis_scale_factor/moe_residual.pt")
        .detach()
        .numpy()
        .flatten()
    )
    moe_residual_x, moe_residual_y = line_graph(moe_residual, label="moe_residual")
    plt.xlim(-0.3, 0.3)
    plt.legend()
    plt.savefig("results/analysis_scale_factor/hist.png")
    plt.close()

    plt.plot(dense_hidden_x, dense_hidden_y, label="dense_hidden")
    plt.fill_between(
        dense_hidden_x, dense_hidden_y, [0] * len(dense_hidden_x), alpha=0.1
    )
    plt.plot(dense_residual_x, dense_residual_y, label="dense_residual")
    plt.fill_between(
        dense_residual_x, dense_residual_y, [0] * len(dense_residual_x), alpha=0.1
    )
    plt.plot(moe_hidden_x, moe_hidden_y, label="moe_hidden")
    plt.fill_between(moe_hidden_x, moe_hidden_y, [0] * len(moe_hidden_x), alpha=0.1)
    plt.plot(moe_residual_x, moe_residual_y, label="moe_residual")
    plt.fill_between(
        moe_residual_x, moe_residual_y, [0] * len(moe_residual_x), alpha=0.1
    )
    plt.xlim(-0.3, 0.3)
    plt.legend()
    plt.show()
    plt.savefig("results/analysis_scale_factor/comparison.png")


if __name__ == "__main__":
    # simulation()
    stat_plot()
