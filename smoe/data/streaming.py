"""
References:
    - https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/packed_dataset.py
    - https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
    - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from smoe.data.aggregation import group_instances
from smoe.utils.io import load_jsonlines, load_jsonlines_iter
from smoe.utils.logging import get_logger
from smoe.utils.random_utils import get_random_string
from smoe.utils.vars import JSONL_DATASET_CACHE_NAME, META_SUFFIX

logger = get_logger(__file__)


class JsonlDataset(IterableDataset):
    def __init__(
        self,
        filepath: str,
        cache_dir: str,
        uid: str = None,
        seed: int = 1227,
        buffer_size: int = 32,
        num_skip: int = None,
        file_start_byte: int = None,
    ) -> None:
        super().__init__()

        if uid:
            self.uid = uid
        else:
            self.uid = f"{Path(self.filepath).stem}-{get_random_string()}"
        self.cache_dir = cache_dir

        self.filepath = filepath
        self.seed = seed
        self.rng = random.Random(seed)
        self.buffer_size = buffer_size
        self.num_skip = num_skip
        self.file_start_byte = file_start_byte
        self.num_yield = 0

        if self.file_start_byte and self.num_skip:
            raise ValueError("Cannot set both `file_start_byte` and `num_skip`")

        self.load_fh = load_jsonlines_iter(
            self.filepath, start_from=self.file_start_byte
        )
        self.buffer = []

    def state_dict(self):
        return {
            "filepath": self.filepath,
            "cache_dir": self.cache_dir,
            "uid": self.uid,
            "seed": self.seed,
            "rng": self.rng.getstate(),
            "num_skip": self.num_skip,
            "file_start_byte": self.file_start_byte,
            "buffer_size": self.buffer_size,
            "num_yield": self.num_yield,
            "load_fh_tell": self.load_fh.tell(),
            "buffer": self.buffer,
        }

    def save_pretrained(self, output_dir: str):
        state_dict = self.state_dict()
        name = JSONL_DATASET_CACHE_NAME.format(self.uid)
        dump_path = Path(output_dir) / name
        torch.save(state_dict, dump_path)
        return str(dump_path)

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        obj = cls(
            state_dict["filepath"],
            state_dict["cache_dir"],
            uid=state_dict["uid"],
            seed=state_dict["seed"],
            buffer_size=state_dict["buffer_size"],
            num_skip=state_dict["num_skip"],
            file_start_byte=state_dict["file_start_byte"],
        )
        obj.rng.setstate(state_dict["rng"])
        obj.num_yield = state_dict["num_yield"]
        obj.buffer = state_dict["buffer"]
        return obj

    @classmethod
    def from_pretrained(cls, state_dict_filepath: str):
        state_dict = torch.load(state_dict_filepath)
        return cls.from_state_dict(state_dict)

    def __iter__(self) -> Iterator:
        self.buffer = []
        for ins in self.load_fh:
            if self.num_skip and self.num_yield < self.num_skip:
                self.num_yield += 1
                continue

            if self.buffer_size <= 1:
                yield ins
                continue

            if len(self.buffer) >= self.buffer_size:
                if len(self.buffer) > 0:
                    self.rng.shuffle(self.buffer)
                    yield from self.buffer
                    self.num_yield += len(self.buffer)
                    self.buffer.clear()

            self.buffer.append(ins)

        # for the last batch < buffer_size
        if len(self.buffer) > 0:
            self.rng.shuffle(self.buffer)
            yield from self.buffer
            self.num_yield += len(self.buffer)
            self.buffer.clear()


class WeightedPackedDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[IterableDataset],
        weights: list[float] = None,
        seed: int = 1227,
    ):
        self.datasets = datasets
        self.weights = weights
        if weights:
            assert len(datasets) == len(weights)
        self.rng = random.Random(seed)

    def __iter__(self):
        while len(self.datasets) > 0:
            candidate_ids = list(range(self.datasets))
            choice = self.rng.choices(candidate_ids, weights=self.weights, k=1)[0]
            try:
                yield next(self.datasets[choice])
            except StopIteration:
                self.datasets.pop(choice)
                if self.weights:
                    self.weights.pop(choice)
                yield from self


class WeightedPackedDatasetBuilder:
    def __init__(
        self,
        filepaths: list[str],
        cache_dir: str,
        resume: bool = False,
        seed: int = 1227,
        buffer_size: int = 32,
    ) -> None:
        self.rng = random.Random(seed)

        self.filepaths = filepaths
        self.rng.shuffle(self.filepaths)
        self.datasets = []

        resumed_path_to_state_dict = {}
        if resume:
            for path in Path(cache_dir).glob(JSONL_DATASET_CACHE_NAME.format("*")):
                state_dict = torch.load(path)
                resumed_path_to_state_dict[state_dict["filepath"]] = state_dict

        for filepath in self.filepaths:
            if filepath in resumed_path_to_state_dict:
                state_dict = resumed_path_to_state_dict[filepath]
                self.datasets.append(JsonlDataset.from_state_dict(state_dict))
            else:
                self.datasets.append(
                    JsonlDataset(
                        filepath,
                        cache_dir,
                        seed=seed,
                        buffer_size=buffer_size,
                    )
                )

    def __iter__(self) -> Iterator:
        for ds in self.datasets:
            yield from ds


class CachedJsonlDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        seed: int = 1227,
        buffer_size: int = 700,
        block_size: int = 2048,
    ):
        super().__init__()
        self.filepath = filepath
        self.rng = random.Random(seed)
        self.buffer_size = buffer_size
        self.block_size = block_size

        dataset = load_jsonlines(self.filepath)
        self.cached = group_instances(dataset, self.block_size)

    def __getitem__(self, index: int):
        return self.cached[index]

    def __len__(self):
        return len(self.cached)


def batchify_loader(dataset: Iterable, batch_size: int, collate_fn: Callable):
    batch = []
    for ins in dataset:
        batch.append(ins)
        if len(batch) >= batch_size:
            yield collate_fn(batch)
            batch.clear()
    if len(batch) > 0:
        yield collate_fn(batch)
        batch.clear()


class BufferAggregation:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    def __call__(self, buffer) -> Any:
        results = buffer
        if self.block_size > 0 and len(buffer) > 0:
            results = group_instances(buffer, self.block_size)
        return results


class PackedJsonlDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        seed: int = 1227,
        buffer_size: int = 200,
        block_size: int = 2048,
        skip_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.rng = random.Random(seed)
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.skip_tokens = skip_tokens

        data_dir_path = Path(data_dir)
        filepaths = sorted(data_dir_path.glob("**/*.jsonl"))
        self.rng.shuffle(filepaths)
        self.filepaths = filepaths
        self.curr_filepath_pointer = -1
        self.consumed_tokens: int = 0

        self.buffer_aggregation = BufferAggregation(self.block_size)

    def next_filepath(self) -> str:
        if len(self.filepaths) == 0:
            raise RuntimeError(f"There's no filepath in {self.data_dir}")

        self.curr_filepath_pointer += 1
        if self.curr_filepath_pointer >= len(self.filepaths):
            self.curr_filepath_pointer = 0

        num_skipped_filepath = 0
        while self.consumed_tokens < self.skip_tokens:
            filepath = self.filepaths[self.curr_filepath_pointer]
            # meta: [[current token number in the whole file, length of the current instance], ...]
            meta: np.ndarray = np.load(filepath + META_SUFFIX)
            curr_filepath_tokens = meta.sum(axis=0)[1]
            if self.consumed_tokens + curr_filepath_tokens > self.skip_tokens:
                break
            self.consumed_tokens += curr_filepath_tokens
            self.curr_filepath_pointer += 1
            if self.curr_filepath_pointer >= len(self.filepaths):
                self.curr_filepath_pointer = 0
            num_skipped_filepath += 1

        if num_skipped_filepath > 0:
            logger.info(
                f"Skip {num_skipped_filepath} files,"
                f" {self.consumed_tokens} tokens,"
                f" remaining {self.skip_tokens - self.consumed_tokens} tokens to skip."
            )

        return self.filepaths[self.curr_filepath_pointer]

    def __iter__(self) -> Iterator:
        filepath = self.next_filepath()
        logger.debug(f"Iter over jsonl file: {filepath}")
        ds = load_jsonlines_iter(filepath)
        # if self.consumed_tokens < self.skip_tokens:
        #     remaining_skip_tokens = self.skip_tokens - self.consumed_tokens
        #     # zhutong: here, the skip method is not perfect since there is batch grouping,
        #     #   and the final token number per instance may be different.
        #     num_skip_lines = (meta[:, 1].cumsum() > remaining_skip_tokens).nonzero()[0][0]
        #     ds.skip_lines(num_skip_lines)
        #     self.consumed_tokens += meta[:num_skip_lines].sum(axis=0)[1]
        for batch in batchify_loader(ds, self.buffer_size, self.buffer_aggregation):
            for ins in batch:
                if self.consumed_tokens >= self.skip_tokens:
                    self.consumed_tokens += len(ins["input_ids"])
                    yield ins

    def state_dict(self):
        return {
            "data_dir": self.data_dir,
            "seed": self.seed,
            "rng": self.rng.getstate(),
            "buffer_size": self.buffer_size,
            "block_size": self.block_size,
            "filepaths": self.filepaths,
            "consumed_tokens": self.consumed_tokens,
        }


class SubDirWeightedPackedJsonlDataset(IterableDataset):
    """
    Example:
        >>> dataset = SubDirWeightedPackedJsonlDataset(
        ...     "/mnt/petrelfs/share_data/redpajama/tokenized",
        ...     weights={
        ...         "en_cc": 0.67,
        ...         "en_c4": 0.15,
        ...         "github": 0.045,
        ...         "en_wikipedia": 0.045,
        ...         "en_book": 0.045,
        ...         "en_arxiv": 0.025,
        ...         "en_stack": 0.02,
        ...     }
        ... )
        >>> for ins in dataset:
        ...     print(ins)

    Inputs:
        dataset_dir: folder structure is:
            task1 dir: 1.jsonl, 2.jsonl, ...
            task2 dir: 1.jsonl, ...
        weights: dirname to sampling weight.
            e.g. {"task1 dir": 0.3, "task2 dir": 0.7}
    """

    def __init__(
        self,
        dataset_dir: str,
        prob_map: dict[str, float] = None,
        seed: int = 1227,
        buffer_size: int = 200,
        block_size: int = 2048,
        skip_tokens: dict = {},
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.dataset_dir_path = Path(dataset_dir)

        task_types = [p.stem for p in self.dataset_dir_path.glob("*") if p.is_dir()]

        if prob_map is None:
            prob_map = {str(task_type): 1.0 for task_type in task_types}
        for task_type in task_types:
            assert task_type in prob_map
        for task_type in prob_map:
            if task_type not in task_types:
                logger.warning(
                    f"Task type {task_type} not found in dataset dir. Skip it."
                )
        self.prob_map = prob_map

        self.consumed_tokens = skip_tokens
        self.task_type_to_dataset = {}
        for task_type in task_types:
            # zhutong: use iter to support next() calling, since the dataset itself
            #          does not implement __next__().
            ds = iter(
                PackedJsonlDataset(
                    str(self.dataset_dir_path.joinpath(task_type)),
                    seed=seed,
                    buffer_size=buffer_size,
                    block_size=block_size,
                    skip_tokens=skip_tokens.get(task_type, 0),
                )
            )
            self.task_type_to_dataset[task_type] = ds

    def skip_tokens(self, skip_tokens: dict):
        for task_type, num_skip_tokens in skip_tokens.items():
            self.task_type_to_dataset[task_type] = iter(
                PackedJsonlDataset(
                    str(self.dataset_dir_path.joinpath(task_type)),
                    seed=self.seed,
                    buffer_size=self.buffer_size,
                    block_size=self.block_size,
                    skip_tokens=skip_tokens.get(task_type, 0),
                )
            )
            if task_type not in self.consumed_tokens:
                self.consumed_tokens[task_type] = 0
            self.consumed_tokens[task_type] += num_skip_tokens

    def __iter__(self) -> Iterator:
        while len(self.task_type_to_dataset) > 0:
            candidate_task_types = list(self.task_type_to_dataset.keys())
            weights = [self.prob_map[task_type] for task_type in candidate_task_types]
            choice = self.rng.choices(candidate_task_types, weights=weights, k=1)[0]
            try:
                ins = next(self.task_type_to_dataset[choice])
                if choice not in self.consumed_tokens:
                    self.consumed_tokens[choice] = 0
                self.consumed_tokens[choice] += len(ins["input_ids"])
                yield ins
            except StopIteration:
                # self.task_type_to_dataset.pop(choice)
                # logger.debug(f"Task type {choice} finished, drop it")
                # yield from self
                return
