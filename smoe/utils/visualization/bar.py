from pathlib import Path

import matplotlib.pyplot as plt


def barh(
    label2num: dict,
    title: str = "No Title",
    save_filepath=None,
    sort_type="label",
    limit=None,
):
    """
    Refers to https://gist.github.com/Spico197/40f0224f9202ef645ac86637a958eaff

    Args:
        sort_type: label or num
    """
    assert sort_type in ["label", "num"]
    if sort_type == "label":
        label2num_sorted = sorted(label2num.items(), key=lambda x: x[0])
    else:
        label2num_sorted = sorted(label2num.items(), key=lambda x: x[1])
    if limit:
        label2num_sorted = label2num_sorted[:limit]
    tot = sum([x[1] for x in label2num_sorted])
    fig = plt.figure(figsize=(16, 9), dpi=350)
    ax = fig.add_subplot(111)
    ax.barh(range(len(label2num_sorted)), [x[1] for x in label2num_sorted], zorder=3)
    ax.set_yticks(range(len(label2num_sorted)))
    ax.set_yticklabels(
        [
            "{} - {} ({:.2f}%)".format(x[0], x[1], float(x[1]) / tot * 100)
            for x in label2num_sorted
        ],
        fontsize=16,
    )
    ax.set_xlabel("Total: {}".format(tot), fontsize=16)
    ax.set_title(title)
    ax.grid(zorder=0)
    plt.rc("axes", axisbelow=True)
    plt.rc("ytick", labelsize=16)
    plt.tight_layout()
    # plt.show()
    if save_filepath:
        Path(save_filepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_filepath)
