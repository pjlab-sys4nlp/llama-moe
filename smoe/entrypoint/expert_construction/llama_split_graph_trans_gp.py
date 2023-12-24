import argparse
import os
import random
import sys
from collections import defaultdict

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpmetised_file_path", type=str, default="./file_path")

    args = parser.parse_args()

    labels = []

    with open(args.gpmetised_file_path) as fin:
        d = defaultdict(list)
        for i, line in enumerate(fin):
            labels.append(int(line.strip()))
            d[labels[-1]].append(i)

    need_move = []

    for i in range(max(d.keys()) + 1):
        if i not in d:
            d[i] = []
    print(len(labels), len(d.keys()))

    num = len(labels) // len(d.keys())
    for k, v in d.items():
        if len(v) > num:
            random.shuffle(v)
            for i in range(num, len(v)):
                need_move.append(v[i])
            d[k] = v[:num]

    print("need_move", need_move)

    random.shuffle(need_move)
    for k, v in d.items():
        if len(v) < num:
            pos = num - len(v)
            v += need_move[:pos]
            need_move = need_move[pos:]
        for x in v:
            labels[x] = k

    vec = os.path.basename(args.gpmetised_file_path).split(".")[:-2]
    target = ".".join(vec)

    save_folder = os.path.join(os.path.dirname(args.gpmetised_file_path), "gp_split")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(labels, os.path.join(save_folder, target))

    from collections import Counter

    print(Counter(labels))
