"""
split files in a folder to separate folders

src: en_arxiv/*
tgt: output/part0, output/part1, ...
"""

from pathlib import Path


def split_files(src_dir, tgt_dir, num_parts):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(src_dir.glob("*.jsonl"))
    num_files = len(filepaths)
    num_files_per_part = num_files // num_parts
    print(f"{src_dir} --> {tgt_dir}")
    print(f"num_files_per_part: {num_files_per_part}")

    for i in range(num_parts):
        start = i * num_files_per_part
        end = (i + 1) * num_files_per_part
        if i == num_parts - 1:
            end = num_files
        print(f"part-{i}, start: {start}, end: {end}")

        part_dir = tgt_dir / f"part-{i:06d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        for j in range(start, end):
            filepath = filepaths[j]
            tgt_filepath = part_dir / filepath.name
            tgt_filepath.symlink_to(filepath)


if __name__ == "__main__":
    for data_type in [
        # "en_arxiv",
        # "en_book",
        # "en_c4",
        "en_cc",
        # "en_stack",
        # "en_wikipedia",
        # "github",
    ]:
        split_files(
            f"/mnt/hwfile/share_data/zhutong/slimpajama_fluency_llama/{data_type}",
            f"/mnt/hwfile/share_data/zhutong/data/slimpajama_fluency_llama_middle_parts/{data_type}",
            30,
        )
    # split_files(
    #     "/mnt/hwfile/share_data/zhutong/slimpajama_fluency_llama/en_arxiv",
    #     "/mnt/hwfile/share_data/zhutong/data/slimpajama_fluency_llama_middle_parts/en_arxiv",
    #     30,
    # )
