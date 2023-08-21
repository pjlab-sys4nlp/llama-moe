import argparse
import os

import pandas as pd


def gather_results(args):
    df = pd.DataFrame(columns=["dataset", "accuracy"])
    for (dir_path, dir_names, file_names) in os.walk(args.save_dir):
        print(dir_path)
        for name in sorted(file_names):
            print(name)
            if name == "all_datasets_0.txt":
                file_path = os.path.join(dir_path, name)
                with open(file_path, "r") as file:
                    for i, line in enumerate(file.readlines()):
                        acc = float(line[17:22])
                        dataset = line[25:-2]
                        if not dataset in df["dataset"].values.tolist():
                            df.loc[i] = [dataset, acc]

            if name == "all_datasets_1.txt":
                file_path = os.path.join(dir_path, name)
                with open(file_path, "r") as file:
                    for i, line in enumerate(file.readlines()):
                        acc = float(line[17:22])
                        dataset = line[25:-2]
                        if not dataset in df["dataset"].values.tolist():
                            df.loc[i + 28] = [dataset, acc]

            if name == "all_datasets_2.txt":
                file_path = os.path.join(dir_path, name)
                with open(file_path, "r") as file:
                    for i, line in enumerate(file.readlines()):
                        acc = float(line[17:22])
                        dataset = line[25:-2]
                        if not dataset in df["dataset"].values.tolist():
                            df.loc[i + 44] = [dataset, acc]

            if name == "all_datasets_3.txt":
                file_path = os.path.join(dir_path, name)
                with open(file_path, "r") as file:
                    for i, line in enumerate(file.readlines()):
                        acc = float(line[17:22])
                        dataset = line[25:-2]
                        if not dataset in df["dataset"].values.tolist():
                            df.loc[i + 57] = [dataset, acc]

        avg_value = float(df["accuracy"].mean())
        df.loc[60] = ["avg_value", avg_value]
        df.to_csv(os.path.join(dir_path, "all_datasets.csv"), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results_moe")
    args = parser.parse_args()
    gather_results(args)
