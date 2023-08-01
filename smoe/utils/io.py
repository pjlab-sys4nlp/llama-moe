import gzip
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
