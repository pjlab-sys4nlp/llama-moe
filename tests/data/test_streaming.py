import tempfile
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from smoe.data.streaming import JsonlDataset, SubDirWeightedPackedJsonlDataset
from smoe.utils.io import load_jsonlines


def test_jsonl_dataset():
    def _get_num_iter(ds):
        num_ins = 0
        for _ in ds:
            num_ins += 1
        return num_ins

    filepath = "/mnt/petrelfs/zhutong/smoe/resources/redpajama/en_arxiv/head2k.jsonl"
    data = load_jsonlines(filepath)

    dataset = JsonlDataset(filepath, buffer_size=16)
    assert len(data) == _get_num_iter(dataset)

    num_skip = 50
    dataset = JsonlDataset(filepath, num_skip=num_skip)
    assert len(data) - num_skip == _get_num_iter(dataset)

    dataset = JsonlDataset(filepath, buffer_size=6)
    num_ins = 0
    for _ in dataset:
        num_ins += 1
        if num_ins == num_skip:
            break
    start_from = dataset.load_fh.tell()
    temp_dir = tempfile.mkdtemp()
    path = dataset.save_pretrained(temp_dir)

    new_dataset = JsonlDataset.from_pretrained(temp_dir)


@pytest.mark.skipif(
    Path("resources/data_test").exists(), reason="Test data dir not found"
)
def test_subdir_weighted_pack():
    from tqdm import tqdm

    dataset = SubDirWeightedPackedJsonlDataset(
        "resources/data_test",
        weights={"en_arxiv": 0.5, "en_book": 0.2, "en_c4": 0.3},
    )
    for _ in tqdm(dataset):
        pass


if __name__ == "__main__":
    # test_jsonl_dataset()
    test_subdir_weighted_pack()
