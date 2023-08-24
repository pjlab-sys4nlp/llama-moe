from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def visualize_expert_load(
    load_sum: np.ndarray,
    layer_idx: int,
    dataset_name: str,
    shape: tuple = (4, 4),
    dtype=int,
):
    path = Path(f"results/expert_load_vis/{dataset_name}_Layer{layer_idx}.png")
    path.parent.mkdir(exist_ok=True, parents=True)

    data = load_sum.reshape(*shape)

    cmap = mpl.colormaps["OrRd"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")

    for i in range(shape[0]):
        for j in range(shape[1]):
            if isinstance(dtype, int):
                ax.text(
                    j, i, f"{data[i, j]:.0f}", ha="center", va="center", color="black"
                )
            else:
                ax.text(
                    j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black"
                )

    ax.set_title(f"{dataset_name} - Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
