import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

from datasets import IterableDataset, load_dataset
from datasets.combine import concatenate_datasets, interleave_datasets
from tqdm import tqdm

from smoe.data.aggregation import group_texts

logger = logging.getLogger(__name__)


def load_streaming_datasets(
    data_dir: str,
    prob_map: dict[str, float] = None,
    num_proc: int = None,
    debug_mode: bool = False,
    block_size: int = 1024,
    split: str = "train",
    verbose: bool = True,
) -> IterableDataset:
    dataset_dir = Path(data_dir)
    files = list(dataset_dir.glob("**/*.jsonl"))
    if debug_mode is True:
        files = [files[0]]

    data_type_to_dataset_list = defaultdict(list)
    grouping_func = partial(group_texts, block_size=block_size)

    fbar = files
    if verbose:
        fbar = tqdm(files, desc="Loading files")
    for filepath in fbar:
        data_type = filepath.parent.stem
        assert (
            data_type in prob_map if prob_map else True
        ), f"{data_type} not in {prob_map.keys()}"
        ds = load_dataset(
            "json",
            data_files=str(filepath),
            num_proc=num_proc,
            streaming=True,
            split=split,
        )
        grouped_datasets = ds.map(
            grouping_func,
            batched=True,
        )
        data_type_to_dataset_list[data_type].append(grouped_datasets)

    datasets_with_diff_types = []
    probs = []
    dbar = None
    if verbose:
        dbar = tqdm(total=len(data_type_to_dataset_list), desc="Concatenating datasets")
    for data_type, datasets in data_type_to_dataset_list.items():
        ds = concatenate_datasets(datasets)
        prob = None
        if prob_map:
            prob = prob_map[data_type]
            probs.append(prob)
        datasets_with_diff_types.append(ds)
        if dbar:
            dbar.update(1)
            dbar.set_postfix({data_type: f"{prob:.3%}%"})

    if len(probs) == 0:
        probs = None
    else:
        sum_probs = sum(probs)
        if sum_probs != 1.0:
            logger.warn(f"Summation of prob_map is {sum_probs}, scaling to 1.0")
            probs = [p / sum_probs for p in probs]

    if verbose:
        logger.info("Grouping datasets")
    lm_datasets = interleave_datasets(datasets_with_diff_types, probs)

    return lm_datasets
