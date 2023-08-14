import json
import os
import warnings

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

"""for hidden feature generation"""


class LineByLineJsonlTextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        num_threads=1,
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

        # lines = lines[:1000]
        self.examples = []
        process_bar = tqdm(
            desc="Reading lines", total=len(lines), leave=False, position=1
        )

        # fmt: off
        if num_threads <= 1:  # use single process
            for line in lines:
                try:
                    # 提前分词，查看分词数量，并以最大长度将文本分块
                    content = json.loads(line)["content"]
                    content_tokenized = tokenizer.tokenize(content)
                    chunk_num = len(content_tokenized) // block_size + 1

                    if chunk_num == 1:  # 只有一块，则直接对原文本编码
                        content_encoding = tokenizer(content, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                        self.examples.append(content_encoding["input_ids"])
                    else:  # 多于一块，对各块分别编码
                        process_bar2 = tqdm(desc="Chunking line", total=chunk_num, leave=False, position=2)
                        for content_tokenized_block in self.split_list_by_n(content_tokenized, block_size):  # chunk content by block_size
                            content_block = " ".join(content_tokenized_block)  # 重新组合为文本形式，以使用tokenizer的encode函数进行自动处理
                            content_encoding = tokenizer(content_block, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                            self.examples.append(content_encoding["input_ids"])
                            process_bar2.update(1)
                        process_bar2.close()
                except Exception:
                    # print("Exception")
                    pass
                process_bar.update(1)
            process_bar.close()

        else:  # don't use this, it is slower than single-processing as data reading is IO consuming instead of CPU consuming
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

    def split_list_by_n(self, list_collection, n):  # 将集合均分，每份n个元素
        for i in range(0, len(list_collection), n):
            yield list_collection[i : i + n]

    def process_line(self, line, tokenizer, block_size):  # 多进程分词函数
        # fmt: off
        if (len(line) > 0 and not line.isspace()):
            # 提前分词，查看分词数量，并以最大长度将文本分块
            content = json.loads(line)["content"]
            content_tokenized = tokenizer.tokenize(content)
            chunk_num = len(content_tokenized) // block_size + 1

            if chunk_num == 1:  # 只有一块，则直接对原文本编码
                content_encoding = tokenizer(content, add_special_tokens=True, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                return [content_encoding["input_ids"]]
            else:  # 多于一块，对各块分别编码
                content_encoding_all = []
                for content_tokenized_block in self.split_list_by_n(content_tokenized, block_size):  # chunk content by block_size
                    content_block = " ".join(content_tokenized_block)  # 重新组合为文本形式，以使用tokenizer的encode函数进行自动处理
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


class ShardDataset(Dataset):  # 从多个数据shard文件中进行数据集读取
    def __init__(
        self,
        path,
        parallel_mode="shards",
        file_load_index_range=None,
        shards_in_memory=8,
    ):  # shards_in_memory只在"shards"模式下有效
        assert parallel_mode in ("shards", "workers")  # 提供两种读取模式，shard并行与worker并行
        self.parallel_mode = parallel_mode

        filename_list = os.listdir(path)
        filename_list.sort()

        if file_load_index_range is None:  # 指定读取文件的范围
            file_load_index_range = (0, len(filename_list) - 1)
        filename_list = filename_list[
            file_load_index_range[0] : file_load_index_range[1]
        ]

        # 适用于单个shard较大的情况
        if (
            self.parallel_mode == "shards"
        ):  # 提前读取shards_in_memory个shard到内存并合并，之后各个workers并行读取内存中的数据
            self.filepath_list = [os.path.join(path, name) for name in filename_list]
            self.chunked_filepath_list = []
            while len(self.filepath_list) > 0:
                self.chunked_filepath_list.append(self.filepath_list[:shards_in_memory])
            self.load_pos = -1
            self.now_epoch = 0
            self.load_shards()

        # 适用于单个shard较小的情况
        elif self.parallel_mode == "workers":  # 不提前读取shard到内存，而是运行时每个worker并行读取shard
            self.filepath_list = [os.path.join(path, name) for name in filename_list]

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

    def load_shards(self):  # "shards"并行模式使用
        object_load_pos = self.now_epoch % len(self.chunked_filepath_list)
        if self.load_pos != object_load_pos:
            self.load_pos = object_load_pos
            self.examples = []
            for filepath in tqdm(
                self.chunked_filepath_list[self.load_pos],
                desc="loading shards",
                leave=False,
            ):  # 单进程读取，使用多进程会由于大量的内存交换而降低速度
                tensor = torch.load(filepath)
                tensor_list = torch.split(
                    tensor.reshape(-1, tensor.shape[-1]), 1, dim=0
                )
                self.examples.extend(tensor_list)
            print("Loaded total {len(self.examples)} examples.")

    def next_epoch(self):  # "shards"并行模式使用
        self.now_epoch += 1
        self.load_shards()


"""for moe gate training"""


class ShardDataset(Dataset):  # 从多个数据shard文件中进行数据集读取
    def __init__(
        self,
        path,
        parallel_mode="shards",
        file_load_index_range=None,
        shards_in_memory=8,
    ):  # shards_in_memory只在"shards"模式下有效
        assert parallel_mode in ("shards", "workers")  # 提供两种读取模式，shard并行与worker并行
        self.parallel_mode = parallel_mode

        filename_list = os.listdir(path)
        filename_list.sort()

        if file_load_index_range is None:  # 指定读取文件的范围
            file_load_index_range = (0, len(filename_list) - 1)
        filename_list = filename_list[
            file_load_index_range[0] : file_load_index_range[1]
        ]

        # 适用于单个shard较大的情况
        if (
            self.parallel_mode == "shards"
        ):  # 提前读取shards_in_memory个shard到内存并合并，之后各个workers并行读取内存中的数据
            self.filepath_list = [os.path.join(path, name) for name in filename_list]
            self.chunked_filepath_list = []
            while len(self.filepath_list) > 0:
                self.chunked_filepath_list.append(self.filepath_list[:shards_in_memory])
            self.load_pos = -1
            self.now_epoch = 0
            self.load_shards()

        # 适用于单个shard较小的情况
        elif self.parallel_mode == "workers":  # 不提前读取shard到内存，而是运行时每个worker并行读取shard
            self.filepath_list = [os.path.join(path, name) for name in filename_list]

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

    def load_shards(self):  # "shards"并行模式使用
        object_load_pos = self.now_epoch % len(self.chunked_filepath_list)
        if self.load_pos != object_load_pos:
            self.load_pos = object_load_pos
            self.examples = []
            for filepath in tqdm(
                self.chunked_filepath_list[self.load_pos],
                desc="loading shards",
                leave=False,
            ):  # 单进程读取，使用多进程会由于大量的内存交换而降低速度
                tensor = torch.load(filepath)
                tensor_list = torch.split(
                    tensor.reshape(-1, tensor.shape[-1]), 1, dim=0
                )
                self.examples.extend(tensor_list)
            print("Loaded total {len(self.examples)} examples.")

    def next_epoch(self):  # "shards"并行模式使用
        self.now_epoch += 1
        self.load_shards()


class ShardDatasetForMoEGate(Dataset):  # 从多个数据shard文件中进行数据集读取
    def __init__(
        self,
        hidden_inputs_path,
        hidden_outputs_path,
        parallel_mode="shards",
        file_load_index_range=None,
        shards_in_memory=8,
    ):
        # fmt: off
        hidden_inputs_filename_list = os.listdir(hidden_inputs_path)
        hidden_inputs_filename_list.sort()
        hidden_outputs_filename_list = os.listdir(hidden_outputs_path)
        hidden_outputs_filename_list.sort()
        assert len(hidden_inputs_filename_list) == len(hidden_outputs_filename_list)

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in ("shards", "workers")  # 提供两种读取模式，shard并行与worker并行

        if file_load_index_range is None:
            file_load_index_range = [0, len(hidden_inputs_filename_list) - 1]  # 未指读取范围，则读取所有文件
        hidden_inputs_filename_list = hidden_inputs_filename_list[file_load_index_range[0]: file_load_index_range[1]]
        hidden_outputs_filename_list = hidden_outputs_filename_list[file_load_index_range[0]: file_load_index_range[1]]

        # 适用于单个shard较大的情况
        if self.parallel_mode == "shards":  # 提前读取shards_in_memory个shard文件到内存后合并，之后各个workers并行读取内存中的数据
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

        # 适用于单个shard较小的情况
        elif self.parallel_mode == "workers":  # 不提前读取shard到内存，而是运行时每个worker并行读取shard文件
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

    def load_shards(self):  # "shards"并行模式下使用
        object_load_pos = self.now_epoch % len(self.chunked_hidden_inputs_filepath_list)

        if self.load_pos != object_load_pos:
            self.load_pos = object_load_pos
            self.hidden_inputs_examples = []
            self.hidden_outputs_examples = []

            # fmt: off
            for filepath in tqdm(self.chunked_hidden_inputs_filepath_list[self.load_pos], desc="loading hidden_inputs shards", leave=False):  # 单进程读取，使用多进程会由于大量的内存交换而降低速度
                tensor = torch.load(filepath, map_location="cpu")
                tensor_list = torch.split(tensor.reshape(-1, tensor.shape[-1]), 1, dim=0)
                self.hidden_inputs_examples.extend(tensor_list)

            for filepath in tqdm(self.chunked_hidden_outputs_filepath_list[self.load_pos], desc="loading hidden_outputs shards", leave=False):  # 单进程读取，使用多进程会由于大量的内存交换而降低速度
                tensor = torch.load(filepath, map_location="cpu")
                tensor_list = torch.split(tensor.reshape(-1, tensor.shape[-1]), 1, dim=0)
                self.hidden_outputs_examples.extend(tensor_list)
            # fmt: on

            print("Loaded total {len(self.hidden_inputs_examples)} examples.")

    def next_epoch(self):  # "shards"并行模式下使用
        self.now_epoch += 1
        self.load_shards()
