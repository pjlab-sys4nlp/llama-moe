import argparse
import os

from smoe.utils.io import compress_png_image


def main(args):
    for dir_path, dir_names, file_names in os.walk(args.root_path):
        for name in file_names:
            if name.endswith(".png"):
                compress_png_image(os.path.join(dir_path, name), print_info=True)
    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str)
    args = parser.parse_args()

    args.root_path = "/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization"

    main(args)
