import gzip
import json
import lzma
import os
import pickle

import torch


def torch_load_template_file(path, template, layer):
    target = os.path.join(path, template.format(layer))
    return torch.load(target)


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

    def tell(self):
        return self.fin.tell()

    def __iter__(self):
        for line in self.fin:
            yield json.loads(line)
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
