from functools import partial

from datasets import IterableDataset, load_dataset

from smoe.data.aggregation import group_texts


def load_cached_dataset(
    filepath: str,
    num_proc: int = None,
    block_size: int = 2048,
    split: str = "train",
) -> IterableDataset:
    grouping_func = partial(group_texts, block_size=block_size)
    ds = load_dataset(
        "json",
        data_files=filepath,
        num_proc=num_proc,
        streaming=True,
        split=split,
    )
    grouped_datasets = ds.map(
        grouping_func,
        batched=True,
    )

    return grouped_datasets
