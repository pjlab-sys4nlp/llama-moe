import argparse

from smoe.utils.moefication.expert_select import summarize_select_result

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    print(args, "\n")

    summarize_select_result(args.result_path, args.save_path)
