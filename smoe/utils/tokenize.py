import argparse
import multiprocessing as mp
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


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
            dformat = "json"
        elif args.format == "txt":
            dformat = "txt"
        else:
            raise ValueError(f"{args.format} format not supported")
        ds = load_dataset(dformat, data_files=input_filepath)
        column_names = ds["train"].column_names
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

        tokenized_ds["train"].to_json(
            output_filepath, lines=True, num_proc=args.num_proc
        )

    if input_path.is_dir():
        input_files = list(input_path.glob(f"*.{args.format}"))
        output_dir = output_path
    else:
        input_files = [input_path]
        output_dir = output_path.parent

    pbar = tqdm(input_files[:2], desc="Tokenization Progress")
    for input_file in pbar:
        out_filename = input_file.stem + ".jsonl"
        output_file = output_dir / out_filename
        input_file = str(input_file)
        output_file = str(output_file)
        pbar.write(f"Processing: {input_file} -> {output_file}")
        _tokenize_and_dump(input_file, output_file)
        pbar.write(f"Finished: {input_file} -> {output_file}")


if __name__ == "__main__":
    tokenize_jsonl()
