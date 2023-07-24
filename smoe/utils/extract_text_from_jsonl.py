"""
Extract texts from jsonlines file.

Example:
    $ python -m smoe.utils.extract_text_from_jsonl -c content -i resources/redpajama/commoncrawl.jsonl -o resources/redpajama-processed/commoncrawl.txt
"""

import argparse
import json


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--column_name", default="content", help="text column name"
    )
    parser.add_argument("-i", "--input_filepath", help="filepath with text to tokenize")
    parser.add_argument("-o", "--output_filepath", help="output filepath")
    args = parser.parse_args()
    return args


def extract_text():
    args = get_parser()

    with open(args.input_filepath, "r", encoding="utf8") as fin:
        with open(args.output_filepath, "w", encoding="utf8") as fout:
            for line in fin:
                ins = json.loads(line)
                text = ins[args.column_name]
                fout.write(f"{text.strip()}\n")


if __name__ == "__main__":
    extract_text()
