import gzip
import lzma
import os

import torch


def torch_load_template_file(path, template, layer):
    target = os.path.join(path, template.format(layer))
    return torch.load(target)


def save_compressed_tensor_7z(tensor, path):  # 7z, too slow
    with lzma.open(path, "wb") as file:
        torch.save(tensor, file)


def load_compressed_tensor_7z(path):  # 7z, too slow
    with lzma.open(path, "rb") as file:
        tensor = torch.load(file)
    return tensor


def save_compressed_tensor_gz(tensor, path, compresslevel=6):  # gz
    with gzip.open(path, "wb", compresslevel=compresslevel) as file:
        torch.save(tensor, file)


def load_compressed_tensor_gz(path):  # gz
    with gzip.open(path, "rb") as file:
        tensor = torch.load(file)
    return tensor
