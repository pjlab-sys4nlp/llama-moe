import os
import shutil
from pathlib import Path

from tqdm import tqdm


def copy_files(src_folder: str, dest_folder: str):
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    files = src_folder.glob("**/*.jsonl")
    for file in tqdm(files):
        dest_file = dest_folder / file.name
        if not dest_file.exists():
            # print(str(file), str(dest_file))
            # shutil.copy2(str(file), str(dest_file))
            # link the file to dest_folder
            # os.link(str(file), str(dest_file))
            os.symlink(str(file), str(dest_file))


if __name__ == "__main__":
    # copy_files(
    #     "/mnt/petrelfs/share_data/quxiaoye/SlimPajama-fluency-processed/c4_split_fluency/",
    #     "/mnt/petrelfs/share_data/quxiaoye/SlimPajama-fluency-processed-agg/en_c4/"
    # )
    for domain in [
        "en_book",
        "en_c4",
        "en_cc",
        "en_arxiv",
        "en_wikipedia",
        "en_stack",
        "github",
    ]:
        copy_files(
            f"/mnt/petrelfs/share_data/zhutong/data/slimpajama_fluency_mistral_middle_parts/{domain}",
            f"/mnt/petrelfs/share_data/zhutong/data/slimpajama_fluency_mistral/{domain}",
        )
