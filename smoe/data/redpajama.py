from collections import defaultdict
from functools import partial
from pathlib import Path

from datasets import IterableDataset, load_dataset
from datasets.combine import concatenate_datasets, interleave_datasets

from smoe.data.aggregation import group_texts


def load_streaming_datasets(
    data_dir: str,
    prob_map: dict[str, float] = None,
    num_proc: int = None,
    debug_mode: bool = False,
    block_size: int = 1024,
) -> IterableDataset:
    dataset_dir = Path(data_dir)
    files = [file.name for file in dataset_dir.glob("**/*.jsonl")]
    if debug_mode is True:
        files = [files[0]]

    data_type_to_dataset_list = defaultdict(list)
    grouping_func = partial(group_texts, block_size=block_size)

    for filepath in files:
        data_type = filepath.parent.stem
        assert data_type in prob_map if prob_map else True
        ds = load_dataset(
            "json", data_files=filepath, num_proc=num_proc, streaming=True
        )
        grouped_datasets = ds.map(
            grouping_func,
            batched=True,
        )
        data_type_to_dataset_list[data_type].append(grouped_datasets)

    datasets_with_diff_types = []
    probs = []
    for data_type, datasets in data_type_to_dataset_list.items():
        ds = concatenate_datasets(datasets)
        if prob_map:
            probs.append(prob_map[data_type])

    if len(probs) == 0:
        probs = None

    lm_datasets = interleave_datasets(datasets_with_diff_types, probs)

    return lm_datasets
