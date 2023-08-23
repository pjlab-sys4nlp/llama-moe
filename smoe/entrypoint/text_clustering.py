import argparse
from pathlib import Path

from smoe.utils.io import dump
from smoe.utils.text_clustering import TextClustering


def main(args):
    data_dir = Path(args.data_dir)
    files = ...

    model = TextClustering(num_experts=args.num_experts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/mnt/petrelfs/share_data/quxiaoye/data/moefication_LLAMA_data",
    )
    parser.add_argument("-n", "--num_experts", type=int, default=16)
    args = parser.parse_args()

    main(args)
