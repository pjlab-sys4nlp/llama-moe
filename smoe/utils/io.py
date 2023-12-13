import gzip
import json
import lzma
import os
import pickle
import shutil

import cv2
import torch


def delete_file_or_dir(dir):
    if os.path.isfile(dir):
        os.remove(dir)
    elif os.path.exists(dir):
        shutil.rmtree(dir)
    else:
        pass


def torch_load_template_file(path, template, layer):
    target = os.path.join(path, template.format(layer))
    return torch.load(target)


def torch_load_template_score_file(path, template, layer):
    score_list = []
    for expert_folder_name in sorted(os.listdir(path)):
        score_file = os.path.join(path, expert_folder_name, template.format(layer))
        score = torch.load(score_file, map_location="cpu")
        score_list.append(score)
    return score_list


def save_compressed_file_7z(tensor, path):  # 7z
    with lzma.open(path, "wb") as file:
        pickle.dump(tensor, file)


def load_compressed_file_7z(path):  # 7z
    with lzma.open(path, "rb") as file:
        data = pickle.load(file)
    return data


def save_compressed_file_gz(tensor, path, compresslevel=6):  # gz
    with gzip.open(path, "wb", compresslevel=compresslevel) as file:
        pickle.dump(tensor, file)


def load_compressed_file_gz(path):  # gz
    with gzip.open(path, "rb") as file:
        data = pickle.load(file)
    return data


class load_jsonlines_iter:
    def __init__(self, filepath, start_from: int = None) -> None:
        self.fin = open(filepath, "r", encoding="utf8")
        if start_from:
            self.fin.seek(start_from, os.SEEK_SET)

    def skip_lines(self, num_skip_lines: int):
        for i, _ in enumerate(self.fin, 1):
            if i == num_skip_lines:
                break

    def tell(self):
        return self.fin.tell()

    def __iter__(self):
        for line in self.fin:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass
        self.fin.close()


def load_jsonlines(filepath):
    data = []
    with open(filepath, "r", encoding="utf8") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def dump_jsonlines(obj, filepath, **kwargs):
    with open(filepath, "w", encoding="utf8") as fout:
        for ins in obj:
            fout.write(f"{json.dumps(ins, ensure_ascii=False, **kwargs)}\n")


def compress_png_image(image_path, print_info=False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if print_info:
        print(f'Done for "{image_path}".')
