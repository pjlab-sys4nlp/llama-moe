import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from smoe.utils.vars import META_SUFFIX


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", required=True, help="tokenizer filepath")
    parser.add_argument(
        "-i", "--input", required=True, help="filepath or dir with jsonl to tokenize"
    )
    parser.add_argument("-o", "--output", required=True, help="output filepath or dir")
    parser.add_argument(
        "-f",
        "--format",
        choices=["jsonl", "txt"],
        required=True,
        help="input file formats: jsonl or txt",
    )
    parser.add_argument(
        "-p",
        "--num_proc",
        default=mp.cpu_count(),
        type=int,
        help="the number of processes for processing",
    )
    parser.add_argument("--use_fast", action="store_true", help="Use fast tokenizer")
    args = parser.parse_args()
    return args


def load_jsonlines(filepath):
    data = []
    with open(filepath, "r", encoding="utf8") as fin:
        for line in tqdm(fin, desc="Loading"):
            ins = json.loads(line)
            if "content" in ins:
                data.append({"content": ins["content"]})
    return data


def load_txt(filepath):
    data = []
    with open(filepath, "r", encoding="utf8") as fin:
        for line in tqdm(fin, desc="Loading"):
            data.append({"content": line.strip()})
    return data


def prepare_meta(jsonl_filepath: str):
    """Prepare metadata for the given jsonl file.

    Args:
        jsonl_filepath (str): tokenized jsonl file path.

    References: https://github.com/InternLM/InternLM/blob/main/tools/tokenizer.py
    License: https://github.com/InternLM/InternLM/blob/main/LICENSE
    """

    meta = []
    cur = 0
    with open(jsonl_filepath, "r", encoding="utf8") as fin:
        for line in fin:
            ins = json.loads(line)
            length = len(ins["input_ids"])
            meta.append((cur, length))
            cur += length

    # define path of the generated meta file
    meta_fp = jsonl_filepath + META_SUFFIX
    # save the generated meta information
    with open(meta_fp, "wb") as f:
        meta = np.array(meta, dtype=np.int32)
        np.save(f, meta)


def tokenize_jsonl():
    args = get_parser()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=args.use_fast)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists():
        assert (input_path.is_dir() and output_path.is_dir()) or (
            input_path.is_file() and output_path.is_file()
        )
    else:
        if input_path.is_dir():
            output_path.mkdir(exist_ok=True, parents=True)
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)

    def _tokenize_and_dump(input_filepath, output_filepath):
        if args.format == "jsonl":
            data = load_jsonlines(input_filepath)
        elif args.format == "txt":
            data = load_txt(input_filepath)
        else:
            raise ValueError(f"{args.format} format not supported")

        ds = Dataset.from_list(data)
        column_names = ds.column_names
        text_column_name = "content" if "content" in column_names else column_names[0]

        def _tokenization_func(examples):
            return {"input_ids": tokenizer(examples[text_column_name])["input_ids"]}

        ds = ds.filter(lambda example: text_column_name in example)
        tokenized_ds = ds.map(
            _tokenization_func,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        tokenized_ds.to_json(output_filepath, lines=True, num_proc=args.num_proc)
        prepare_meta(output_filepath)

    if input_path.is_dir():
        input_files = list(input_path.glob(f"*.{args.format}"))
        output_dir = output_path
    else:
        input_files = [input_path]
        output_dir = output_path.parent

    pbar = tqdm(input_files, desc="Tokenization Progress")
    for input_file in pbar:
        out_filename = input_file.stem + ".jsonl"
        output_file = output_dir / out_filename
        input_file = str(input_file)
        output_file = str(output_file)
        pbar.write(f"Processing: {input_file} -> {output_file}")
        _tokenize_and_dump(input_file, output_file)
        pbar.write(f"Finished: {input_file} -> {output_file}")


def update_meta_without_tokenization(data_dir: str):
    """Update meta information for the given data directory.

    Args:
        data_dir (str): data directory path.
    """
    folder = Path(data_dir)
    jsonl_filepaths = [str(p) for p in folder.glob("**/*.jsonl")]
    sub_dirs = [str(p) for p in folder.glob("*")]
    print(f"Sub directories (tot={len(sub_dirs)}): {sub_dirs}")
    print(f"#Total Jsonl files: {len(jsonl_filepaths)}")

    mp.cpu_count()
    bar = tqdm(range(len(jsonl_filepaths)), desc="Preparing meta")
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in pool.imap_unordered(prepare_meta, jsonl_filepaths):
            bar.update()


if __name__ == "__main__":
    tokenize_jsonl()

    # # uncomment and run: srun -p MoE -c 16 python -m smoe.utils.tokenize
    # update_meta_without_tokenization(
    #     "/mnt/petrelfs/share_data/quxiaoye/SlimPajama_processed"
    # )
