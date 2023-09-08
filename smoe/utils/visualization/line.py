import matplotlib.pyplot as plt


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
    plt.close()
