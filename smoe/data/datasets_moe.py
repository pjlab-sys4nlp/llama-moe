import json
import math
import os
import warnings

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from smoe.utils.operations.operation_list import split_list_with_yield

"""for hidden feature generation"""


class LineByLineJsonlTextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        data_index_range: tuple = None,
        num_threads: int = 1,
    ):
        """numthreads should be set <=1, otherwise it will slow down the reading process by ~4 times"""
        if num_threads > 1:
            warnings.warn(
                "num_threads should be set <=1, otherwise it will slow down the reading"
                " process by ~4 times!"
            )

        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        with open(file_path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        if data_index_range is not None:
            lines = lines[data_index_range[0] : data_index_range[1]]
        self.examples = []
        process_bar = tqdm(
            desc="Reading lines", total=len(lines), leave=False, position=1
        )

        # fmt: off
        if num_threads <= 1:  # use single process
            for line in lines:
                try:
                    # Tokenize in advance, check the number of tokens, and split the text into chunks of the maximum length
                    content = json.loads(line)["content"]
                    content_tokenized = tokenizer.tokenize(content)
                    chunk_num = len(content_tokenized) // block_size + 1

                    if chunk_num == 1:  # If there's only one chunk, encode the original text directly
                        content_encoding = tokenizer(content, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                        self.examples.append(content_encoding)
                    else:  # If there are multiple chunks, encode each chunk separately
                        process_bar2 = tqdm(desc="Chunking line", total=chunk_num, leave=False, position=2)
                        for content_tokenized_block in split_list_with_yield(content_tokenized, block_size):  # chunk content by block_size
                            content_block = tokenizer.convert_tokens_to_string(content_tokenized_block)  # Reassemble into text format for tokenizer's encode function
                            content_encoding = tokenizer(content_block, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                            self.examples.append(content_encoding)  # dict: {"input_ids":... , "attention_mask":...}
                            process_bar2.update(1)
                        process_bar2.close()
                except Exception:
                    # print("Exception")
                    pass
                process_bar.update(1)
            process_bar.close()

        else:  # don't use this, it is slower than single-processing as data reading is IO bounded instead of CPU bounded
            from pebble import ProcessExpired, ProcessPool
            with ProcessPool(max_workers=num_threads) as pool:
                future = pool.map(self.process_line, lines, [tokenizer] * len(lines), [block_size] * len(lines))
                iterator = future.result()
                while True:
                    try:
                        input_ids = next(iterator)
                        if input_ids is not None:
                            for chunk in input_ids:
                                self.examples.append(chunk)
                        process_bar.update(1)
                    except StopIteration:
                        process_bar.close()
                        break
                    except TimeoutError:
                        print("TimeoutError")
                    except ProcessExpired:
                        print("ProcessExpired")
                    except Exception:
                        print("Exception")
        # fmt: on

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def process_line(
        self, line, tokenizer, block_size
    ):  # Multi-thread tokenization function
        # fmt: off
        if len(line) > 0 and not line.isspace():
            # Tokenize in advance, check the number of tokens, and split the text into chunks of the maximum length
            content = json.loads(line)["content"]
            content_tokenized = tokenizer.tokenize(content)
            chunk_num = len(content_tokenized) // block_size + 1

            if chunk_num == 1:  # If there's only one chunk, encode the original text directly
                content_encoding = tokenizer(content, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                return [content_encoding["input_ids"]]
            else:  # If there are multiple chunks, encode each chunk separately
                content_encoding_all = []
                for content_tokenized_block in split_list_with_yield(content_tokenized, block_size):  # chunk content by block_size
                    content_block = " ".join(content_tokenized_block)  # Reassemble into text format for tokenizer's encode function
                    content_encoding = tokenizer(content_block, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                    content_encoding_all.append(content_encoding["input_ids"])
                return content_encoding_all
        else:
            return None
        # fmt: on


class CommonDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


"""for moe graph split"""


class ShardDataset(Dataset):  # Read datasets from multiple shard files
    def __init__(
        self,
        path,
        parallel_mode="shards",
        data_use_percent=1.0,
        file_load_index_range=(0.0, 1.0),
        shards_in_memory=8,  # Only effective in "shards" mode
    ):
        # fmt: off
        assert parallel_mode in ("shards", "workers")  # Two modes of reading provided: shard parallel and worker parallel
        self.parallel_mode = parallel_mode

        filename_list = os.listdir(path)
        filename_list.sort()

        # Specify the proportion of files to read
        filename_list = filename_list[:math.floor(len(filename_list) * data_use_percent)]

        # Specify the range of files to read
        filename_list = filename_list[math.floor(len(filename_list) * file_load_index_range[0]):
                                      math.floor(len(filename_list) * file_load_index_range[1])]

        # Suitable for cases where individual shards are large
        if self.parallel_mode == "shards":  # Preload shards_in_memory shards into memory and merge; then workers read from memory in parallel
            self.filepath_list = [os.path.join(path, name) for name in filename_list]
            self.chunked_filepath_list = []
            while len(self.filepath_list) > 0:
                self.chunked_filepath_list.append(self.filepath_list[:shards_in_memory])
            self.load_pos = -1
            self.now_epoch = 0
            self.load_shards()

        # Suitable for cases where individual shards are small
        elif self.parallel_mode == "workers":  # No preloading of shards into memory; workers read shards in parallel during runtime
            self.filepath_list = [os.path.join(path, name) for name in filename_list]
        # fmt: on

    def __len__(self):
        if self.parallel_mode == "shards":
            return len(self.examples)

        elif self.parallel_mode == "workers":
            return len(self.filepath_list)

    def __getitem__(self, i):
        if self.parallel_mode == "shards":
            return self.examples[i]

        elif self.parallel_mode == "workers":
            return torch.load(self.filepath_list[i])

    def load_shards(self):  # Used in "shards" parallel mode
        object_load_pos = self.now_epoch % len(self.chunked_filepath_list)
        if self.load_pos != object_load_pos:
            self.load_pos = object_load_pos
            self.examples = []

            # fmt: off
            for filepath in tqdm(self.chunked_filepath_list[self.load_pos], desc="loading shards", leave=False, ):  # Single-threaded read; multi-threading would slow down due to heavy memory exchange
                tensor = torch.load(filepath)
                tensor_list = torch.split(tensor.reshape(-1, tensor.shape[-1]), 1, dim=0)
                self.examples.extend(tensor_list)
            print("Loaded total {len(self.examples)} examples.")
            # fmt: on

    def next_epoch(self):  # Used in "shards" parallel mode
        self.now_epoch += 1
        self.load_shards()


"""for moe gate training"""


class ShardDatasetForMoEGate(
    Dataset
):  # Read datasets from multiple shard files for MoE Gate training
    def __init__(
        self,
        hidden_inputs_path,
        hidden_outputs_path,
        parallel_mode="shards",
        data_use_percent=1.0,
        file_load_index_range=(0.0, 1.0),
        shards_in_memory=8,  # Only effective in "shards" mode
    ):
        # fmt: off
        hidden_inputs_filename_list = os.listdir(hidden_inputs_path)
        hidden_inputs_filename_list.sort()
        hidden_outputs_filename_list = os.listdir(hidden_outputs_path)
        hidden_outputs_filename_list.sort()
        assert len(hidden_inputs_filename_list) == len(hidden_outputs_filename_list)

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in ("shards", "workers")  # Two modes of reading provided: shard parallel and worker parallel

        # Specify the proportion of files to read
        hidden_inputs_filename_list = hidden_inputs_filename_list[:math.floor(len(hidden_inputs_filename_list) * data_use_percent)]
        hidden_outputs_filename_list = hidden_outputs_filename_list[:math.floor(len(hidden_outputs_filename_list) * data_use_percent)]

        # Specify the range of files to read
        hidden_inputs_filename_list = hidden_inputs_filename_list[math.floor(len(hidden_inputs_filename_list) * file_load_index_range[0]):
                                                                  math.floor(len(hidden_inputs_filename_list) * file_load_index_range[1])]
        hidden_outputs_filename_list = hidden_outputs_filename_list[math.floor(len(hidden_outputs_filename_list) * file_load_index_range[0]):
                                                                    math.floor(len(hidden_outputs_filename_list) * file_load_index_range[1])]

        # Suitable for cases where individual shards are large
        if self.parallel_mode == "shards":  # Preload shards_in_memory shard files into memory and merge; then workers read from memory in parallel
            hidden_inputs_filepath_list = [os.path.join(hidden_inputs_path, name) for name in hidden_inputs_filename_list]
            hidden_outputs_filepath_list = [os.path.join(hidden_outputs_path, name) for name in hidden_outputs_filename_list]

            self.chunked_hidden_inputs_filepath_list = []
            self.chunked_hidden_outputs_filepath_list = []

            while len(hidden_inputs_filepath_list) > 0:
                self.chunked_hidden_inputs_filepath_list.append(hidden_inputs_filepath_list[:shards_in_memory])
                self.chunked_hidden_outputs_filepath_list.append(hidden_outputs_filepath_list[:shards_in_memory])
                hidden_inputs_filepath_list = hidden_inputs_filepath_list[shards_in_memory:]
                hidden_outputs_filepath_list = hidden_outputs_filepath_list[shards_in_memory:]

            self.load_pos = -1
            self.now_epoch = 0
            self.load_shards()

        # Suitable for cases where individual shards are small
        elif self.parallel_mode == "workers":  # No preloading of shards into memory; workers read shards in parallel during runtime
            self.hidden_inputs_filepath_list = [os.path.join(hidden_inputs_path, name) for name in hidden_inputs_filename_list]
            self.hidden_outputs_filepath_list = [os.path.join(hidden_outputs_path, name) for name in hidden_outputs_filename_list]
        # fmt: on

    def __len__(self):
        if self.parallel_mode == "shards":
            return len(self.hidden_inputs_examples)

        elif self.parallel_mode == "workers":
            return len(self.hidden_inputs_filepath_list)

    def __getitem__(self, i):
        if self.parallel_mode == "shards":
            return self.hidden_inputs_examples[i], self.hidden_outputs_examples[i]

        elif self.parallel_mode == "workers":
            hidden_inputs = torch.load(
                self.hidden_inputs_filepath_list[i], map_location="cpu"
            )
            hidden_outputs = torch.load(
                self.hidden_outputs_filepath_list[i], map_location="cpu"
            )
            return hidden_inputs, hidden_outputs

    def load_shards(self):  # Used in "shards" parallel mode
        object_load_pos = self.now_epoch % len(self.chunked_hidden_inputs_filepath_list)

        if self.load_pos != object_load_pos:
            self.load_pos = object_load_pos
            self.hidden_inputs_examples = []
            self.hidden_outputs_examples = []

            # fmt: off
            for filepath in tqdm(self.chunked_hidden_inputs_filepath_list[self.load_pos], desc="loading hidden_inputs shards", leave=False):  # Single-threaded read; multi-threading would slow down due to heavy memory exchange
                tensor = torch.load(filepath, map_location="cpu")
                tensor_list = torch.split(tensor.reshape(-1, tensor.shape[-1]), 1, dim=0)
                self.hidden_inputs_examples.extend(tensor_list)

            for filepath in tqdm(self.chunked_hidden_outputs_filepath_list[self.load_pos], desc="loading hidden_outputs shards", leave=False):  # Single-threaded read; multi-threading would slow down due to heavy memory exchange
                tensor = torch.load(filepath, map_location="cpu")
                tensor_list = torch.split(tensor.reshape(-1, tensor.shape[-1]), 1, dim=0)
                self.hidden_outputs_examples.extend(tensor_list)
            # fmt: on

            print("Loaded total {len(self.hidden_inputs_examples)} examples.")

    def next_epoch(self):  # Used in "shards" parallel mode
        self.now_epoch += 1
        self.load_shards()
