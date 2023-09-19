"""
srun -p MoE -n 1 -N 1 --mem 128G python -m smoe.entrypoint.text_clustering --do_train --do_eval -n 16 -m outputs/clustering -o resources/clustering_samples
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from smoe.utils.io import load_jsonlines_iter
from smoe.utils.text_clustering import TextClustering

logger = logging.getLogger(__name__)


def main(args):
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.jsonl"))

    if args.do_train:
        logger.info("Loading contents")
        contents = []
        for file in files:
            for ins in load_jsonlines_iter(file):
                contents.append(ins["content"])

        model = TextClustering(num_clusters=args.num_clusters)
        logger.info("Fitting model")
        model.fit(contents)
        logger.info("Saving model")
        model.save_pretrained(args.model_dir)

    if args.do_eval:
        logger.info("Loading model")
        model = TextClustering.from_pretrained(args.model_dir)

        logger.info("Loading contents")
        instances = []
        for file in files:
            for i, ins in enumerate(load_jsonlines_iter(file)):
                instances.append(
                    {"content": ins["content"], "id": i, "file": file.name}
                )
        logger.info("Predicting")
        labels = model.predict([ins["content"] for ins in instances])

        logger.info("Dumping results")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        cluster_to_num = defaultdict(lambda: 0)
        cluster_to_fout = {
            i: open(out_dir / f"{i}.jsonl", "w") for i in range(args.num_clusters)
        }
        for ins, label in zip(instances, labels):
            cluster_to_fout[label].write(f"{json.dumps(ins, ensure_ascii=False)}\n")
            cluster_to_num[label] += 1

        for fp in cluster_to_fout.values():
            fp.close()

        logger.info(f"Done: {dict(cluster_to_num)}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/mnt/petrelfs/share_data/quxiaoye/data/moefication_LLAMA_data",
    )
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-n", "--num_clusters", type=int, default=16)
    parser.add_argument("-m", "--model_dir", type=str)
    args = parser.parse_args()

    main(args)
