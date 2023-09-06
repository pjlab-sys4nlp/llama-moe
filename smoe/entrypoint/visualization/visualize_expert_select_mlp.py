import argparse

from smoe.utils.visualization.visualize import visualize_expert_select_mlp

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--proj_type', type=str)

    args = parser.parse_args()
    print(args, "\n")

    visualize_expert_select_mlp(args.result_path, args.save_path, args.proj_type)
