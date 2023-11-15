import statistics as sts
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

xs = []
ys = []


def kaiming_init(*size):
    tensor = torch.randn(*size)
    nn.init.kaiming_normal_(tensor, mode="fan_out")
    return tensor


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
