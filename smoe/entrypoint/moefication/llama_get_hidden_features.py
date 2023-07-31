import argparse
import os
import random
import types

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP

from smoe.data.datasets_moefication import CommonDataset, LineByLineJsonlTextDataset


# fmt: off
def change_forward(llama_model, device_id, save_path, template, save_interval=1):
    def _forward(self, x):  # new forward function with hidden_states recording
        self.now_epoch += 1

        self.hidden_inputs.append(x.detach().half())

        if self.now_epoch % self.save_interval == (self.save_interval - 1):
            save_path = os.path.join(self.save_path_hidden_inputs, str(self.device_id) + "_" + str(self.now_epoch // self.save_interval) + ".pth")
            torch.save(torch.cat(self.hidden_inputs, dim=0).reshape(-1, self.hidden_dim).half().cpu(), save_path, pickle_protocol=4)
            self.hidden_inputs = []

        gate_proj_output = self.act_fn(self.gate_proj(x))
        up_proj_output = self.up_proj(x)
        gate_up_mm_output = gate_proj_output * up_proj_output
        down_proj_output = self.down_proj(gate_up_mm_output)

        if "gate_proj" in template:
            self.hidden_outputs.append(gate_proj_output.detach().half())
        elif "up_proj" in template:
            self.hidden_outputs.append(gate_up_mm_output.detach().half())

        if self.now_epoch % self.save_interval == (self.save_interval - 1):
            save_path = os.path.join(self.save_path_hidden_outputs, str(self.device_id) + "_" + str(self.now_epoch // self.save_interval) + ".pth")
            torch.save(torch.cat(self.hidden_outputs, dim=0).reshape(-1, self.hidden_neurons).half().cpu(), save_path, pickle_protocol=4)
            self.hidden_outputs = []

        return down_proj_output

    for layer_idx, layer in enumerate(llama_model.layers):  # locate block by the name template
        mlp = layer.mlp
        assert type(mlp) == LlamaMLP

        mlp.forward = types.MethodType(_forward, mlp)  # change forward function for LlamaMLP
        mlp.hidden_inputs = []
        mlp.hidden_outputs = []

        mlp.device_id = device_id
        mlp.layer_idx = layer_idx
        mlp.now_epoch = -1
        mlp.hidden_dim = llama_model.config.hidden_size
        mlp.hidden_neurons = llama_model.config.intermediate_size
        mlp.save_path_hidden_inputs = os.path.join(save_path, "hidden_inputs", template.format(layer_idx))
        if "gate_proj" in template:
            mlp.save_path_hidden_outputs = os.path.join(save_path, "hidden_gate_outputs", template.format(layer_idx))
        elif "up_proj" in template:
            mlp.save_path_hidden_outputs = os.path.join(save_path, "hidden_up_outputs", template.format(layer_idx))
        mlp.save_interval = save_interval

        if not os.path.exists(mlp.save_path_hidden_inputs):
            os.makedirs(mlp.save_path_hidden_inputs)
        if not os.path.exists(mlp.save_path_hidden_outputs):
            os.makedirs(mlp.save_path_hidden_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--train_data_path', type=str, default="/home/dongdz/workspace/moefication/llama_data/")
    parser.add_argument('--train_data_cache_path', type=str, default="/home/dongdz/workspace/moefication/llama_data_cache/")
    parser.add_argument('--save_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/")
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--data_use_percent', type=float, default=0.01)  # 所有数据集中数据的使用比例，用于调节训练使用的数据量
    parser.add_argument('--batch_size', type=int, default=4)  # 单次evaluate的batch_size
    parser.add_argument('--save_interval', type=int, default=1)  # 保存参数的batch间隔，调大会影响显存占用，但可以减少保存的文件个数

    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, os.path.split(args.model_path)[1] + "-Hidden-Features")

    print("cuda is_available: " + str(torch.cuda.is_available()))
    dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    """load tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    """prepare datasets"""
    dataset_weight = {
        "commoncrawl": 670,
        "c4": 150,
        "github": 45,
        "wikipedia": 45,
        "books": 45,
        "arxiv": 25,
        "stackexchange": 20,
    }

    # re-balance weights according to the max weight
    weight_max = max(dataset_weight.values())
    for key in dataset_weight.keys():
        dataset_weight[key] = dataset_weight[key] / weight_max
    print(dataset_weight)

    # read datasets
    all_datasets = {}
    for key in tqdm(dataset_weight.keys()):
        for file_name in os.listdir(args.train_data_path):
            if key in file_name and file_name.endswith(".jsonl"):
                cached_file_path = os.path.join(args.train_data_cache_path, key + "_cached.pth")
                if os.path.exists(cached_file_path):
                    print("\nReading dataset \"" + key + "\" from cached file \"" + cached_file_path + "\"...")
                    all_datasets[key] = CommonDataset(torch.load(cached_file_path))
                else:
                    raw_file_path = os.path.join(args.train_data_path, file_name)
                    print("\nReading dataset \"" + key + "\" from raw file \"" + raw_file_path + "\"...")
                    all_datasets[key] = LineByLineJsonlTextDataset(tokenizer, file_path=raw_file_path, block_size=2048)
                    if not os.path.exists(args.train_data_cache_path):
                        os.makedirs(args.train_data_cache_path)
                    with open(cached_file_path, "wb") as file:
                        torch.save(all_datasets[key].examples, os.path.join(args.train_data_cache_path, key + "_cached.pth"))
                print("Dataset " + key + ": " + str(sum([torch.sum(all_datasets[key][i] != 2).item() for i in range(len(all_datasets[key]))])) + " total tokens.")  # 统计非padding的token数量

    # scale datasets by weights
    max_available_num = {}  # the maximum number of batches each dataset can provide

    for key in dataset_weight.keys():  # assume this key is 100% used, O(n^2) time complexity
        this_available_num = {key: len(all_datasets[key])}

        for check_key in dataset_weight.keys():  # check if other key can provide enough number of batches
            if key == check_key:
                continue
            if int(len(all_datasets[key]) * (dataset_weight[check_key] / dataset_weight[key])) > len(all_datasets[check_key]):  # the check_key cannot provide enough data
                this_available_num = None  # not satisfied
                break
            else:  # this check_key is satisfied
                this_available_num[check_key] = int(len(all_datasets[key]) * (dataset_weight[check_key] / dataset_weight[key]))

        if this_available_num is not None:
            if len(max_available_num) == 0:
                max_available_num = this_available_num
            else:
                for key2 in dataset_weight.keys():  # update for all keys
                    if this_available_num[key2] > max_available_num[key2]:
                        max_available_num[key2] = this_available_num[key2]

    random.seed(0)
    for key in dataset_weight.keys():  # reset the number of examples by max_available_num and data_use_percent
        all_datasets[key].examples = random.sample(all_datasets[key].examples, int(max_available_num[key] * args.data_use_percent))  # 按照batch分配，由于padding的存在，token的比例会有误差

    # split training set and validation set
    training_dataset = []
    for key in dataset_weight.keys():
        training_dataset.append(CommonDataset(all_datasets[key].examples))
        print("Dataset " + key + ": " + str(sum([torch.sum(training_dataset[-1][i] != 2).item() for i in range(len(training_dataset[-1]))])) + " training tokens.")  # 统计非padding的token数量
    training_dataset = ConcatDataset(training_dataset)

    """prepare dataloader"""
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, rank=args.local_rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    print("All Datasets: " + str(sum([torch.sum(training_dataset[i] != 2).item() for i in range(len(training_dataset))])) + " training tokens.")
    iter_train = iter(train_loader)

    """load model"""
    print("Loading llama model...")
    model = LlamaForCausalLM.from_pretrained(args.model_path).model
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    change_forward(model, args.local_rank, args.save_path, args.template, save_interval=args.save_interval)

    with torch.cuda.device("cuda:" + str(args.local_rank)):
        print("Used GPU memory (GPU " + str(args.local_rank) + ") (Model-CPU): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

    model.cuda(args.local_rank)

    with torch.cuda.device("cuda:" + str(args.local_rank)):
        print("Used GPU memory (GPU " + str(args.local_rank) + ") (Model-Cuda): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    with torch.cuda.device("cuda:" + str(args.local_rank)):
        print("Used GPU memory (GPU " + str(args.local_rank) + ") (Model-DDP): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

    """evaluation"""
    print("Start evaluation...")
    model.eval()
    process_bar2 = tqdm(range(len(train_loader)), desc="forward step", position=0, leave=True)
    for train_step in process_bar2:
        train_sampler.set_epoch(train_step)
        train_batch = next(iter_train)
        train_batch = train_batch.cuda(args.local_rank, non_blocking=True)

        with torch.no_grad():
            model(train_batch)

        process_bar2.update(1)
    process_bar2.close()
    print("Done.")
