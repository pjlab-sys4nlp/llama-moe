import os
import pickle
import random
import types
from collections import Counter

import numpy as np
import sklearn
import torch
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaMLP

from smoe.data.datasets_moefication import ShardDataset
from smoe.utils.change_llama_forward import forward_llama_mlp_with_backward_hook_bug_fix
from smoe.utils.kernel_function import pass_kernel_function
from smoe.utils.moefication.k_means_constrained_cos import KMeansConstrainedCos


def load_ffn_weight(model, template, layer):
    key = template.format(layer)
    return model[key].numpy()


class LayerSplit:
    def __init__(self, config, template, layer):
        self.config = config
        self.template = template
        self.layer = layer

    def save(self):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        filename = os.path.join(self.config.save_path, self.template.format(self.layer))
        torch.save(self.labels, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Expert indices for layer {self.layer} saved to "{filename}".')

    def cnt(self):
        print(Counter(self.labels))


class RandomSplit(LayerSplit):
    def __init__(self, config, model_config, template, layer):
        super().__init__(config, template, layer)
        self.model_config = model_config
        self.neuron_num = model_config.intermediate_size
        self.split_size = self.neuron_num // self.config.num_experts

    def split(self):
        self.labels = np.arange(0, self.config.num_experts, dtype=int).tolist()  # list
        self.labels = self.labels * self.split_size
        random.shuffle(self.labels)


class ClusteringSplit(LayerSplit):
    def __init__(self, config, model, template, layer, distance="l2"):
        super().__init__(config, template, layer)
        self.type = "split_clustering"
        self.distance = distance
        self.model = model
        self.model_dict = model.state_dict()

    def load_param(self):
        self.ffn_weight = load_ffn_weight(self.model_dict, self.template, self.layer)
        self.neuron_num = self.ffn_weight.shape[0]
        self.split_size = self.neuron_num // self.config.num_experts
        assert self.split_size * self.config.num_experts == self.neuron_num

    def split(self, cpu_threads=-1):
        self.load_param()
        # ffn_weight_norm = sklearn.preprocessing.normalize(self.ffn_weight)
        ffn_weight_norm = Normalizer().transform(self.ffn_weight)

        if self.distance.lower() == "l2":
            kmeans = KMeansConstrained(
                n_clusters=self.config.num_experts,
                size_min=self.split_size,
                size_max=self.split_size,
                max_iter=500,
                random_state=0,
                n_jobs=cpu_threads,
                verbose=True,
            ).fit(ffn_weight_norm, None)

        elif self.distance.lower() == "cos":
            kmeans = KMeansConstrainedCos(
                n_clusters=self.config.num_experts,
                size_min=self.split_size,
                size_max=self.split_size,
                max_iter=500,
                random_state=0,
                n_jobs=cpu_threads,
                verbose=True,
            ).fit(ffn_weight_norm, None)

        self.labels = [x for x in kmeans.labels_]


class GraphSplit(LayerSplit):
    def __init__(self, config, model, template, layer):
        super().__init__(config, template, layer)
        self.device = "cpu"
        self.config = config
        self.threshold = config.threshold  # 1 # threshold
        self.template = template
        self.save_path = config.save_path
        self.model = model
        self.model_dict = model.state_dict()
        self.metric = config.metric
        hidden_features_path = config.hidden_features_path
        if "gate_proj" in template:
            hidden_outputs_path = os.path.join(
                hidden_features_path, "hidden_gate_outputs", "layer" + str(layer)
            )
        elif "up_proj" in template:
            hidden_outputs_path = os.path.join(
                hidden_features_path, "hidden_up_outputs", "layer" + str(layer)
            )
        else:
            raise ValueError

        dataset = ShardDataset(hidden_outputs_path, parallel_mode="workers")
        self.dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
        )

    def load_param(self):
        self.ffn_weight = load_ffn_weight(self.model_dict, self.template, self.layer)
        self.neuron_num = self.ffn_weight.shape[0]
        self.split_size = self.neuron_num // self.config.num_experts
        assert self.split_size * self.config.num_experts == self.neuron_num

    def split_and_save(self):
        self.load_param()
        ffn = torch.tensor(self.ffn_weight)

        cnt = 0
        adj = torch.zeros(ffn.shape[0], ffn.shape[0], device=self.device).float()
        ffn = torch.tensor(ffn).to(self.device).transpose(0, 1)
        iter_train = iter(self.dataloader)

        for indx in tqdm(range(len(self.dataloader))):
            hidden = next(iter_train)
            hidden = hidden.to(self.device).float()
            # res = hidden * hidden # 8192
            res = pass_kernel_function(hidden, self.metric)
            res = torch.clamp(
                torch.bmm(res.transpose(1, 2), res).sum(0), max=self.threshold
            )
            print(self.layer, indx, torch.nonzero(torch.isnan(res)), flush=True)
            # tqdm.write(f"{self.layer} {indx} {torch.nonzero(torch.isnan(res))}")
            adj = adj + res

        del hidden

        adj = adj.cpu().numpy()
        target = os.path.join(self.save_path, self.template.format(self.layer))

        threshold = 0
        pos = 10
        while threshold == 0:
            assert pos != 110
            threshold = np.percentile(adj.reshape(-1), pos)
            pos += 10
        print("threshold", threshold, pos, adj.max())
        threshold = threshold * 0.99
        adj /= threshold

        with open(target, "w") as fout:
            edges = 0
            for i in range(adj.shape[0]):
                cnt = 0
                for j in range(adj.shape[1]):
                    if i == j or adj[i, j] < 1:
                        pass
                    else:
                        cnt += 1
                edges += cnt
            assert edges > 0
            fout.write("{} {} {}\n".format(adj.shape[0], edges // 2, "001"))
            for i in range(adj.shape[0]):
                vec = []
                for j in range(adj.shape[1]):
                    if i == j or adj[i, j] < 1:
                        pass
                    else:
                        val = int(adj[i, j])
                        vec.append([j + 1, val])
                fout.write(" ".join(["{} {}".format(x[0], x[1]) for x in vec]) + "\n")


class GradientSplitGetGrads:
    # fmt: off
    """
    SNIP: Single-shot Network Pruning based on Connection Sensitivity (ICLR 2019)
    """

    def __init__(self, config, trainer, accumulate_level="total", kernel="l1_norm", device="cpu"):
        self.config = config
        self.trainer = trainer

        self.accumulate_level = accumulate_level
        self.kernel = kernel
        self.device = device

        self.layer_num = self.trainer.model.config.num_hidden_layers
        self.neuron_num = self.trainer.model.config.intermediate_size

        self.sample_count = 0
        self.up_proj_grads = {}
        self.gate_proj_grads = {}
        for i in range(self.layer_num):
            self.up_proj_grads[i] = torch.zeros((self.neuron_num,), device=self.device)
            self.gate_proj_grads[i] = torch.zeros((self.neuron_num,), device=self.device)

    def _backward_hook_up_proj(self, module, grad_in, grad_out):
        if module.add_batch_size:
            batch_size = grad_in[0].shape[0]
            self.sample_count += batch_size - 1  # gradient of the last token in the batch is 0
            # print("sample_count:", self.sample_count)

        # if self.device == "cuda:0":
        #     with torch.cuda.device("cuda:0"):
        #         print("Used GPU memory (GPU 0): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")
        #     print("up", "grad_out", len(grad_out), [grad_out[i].shape if grad_out[i] is not None else None for i in range(len(grad_out))], grad_out, sep='\n')
        #     print("up", "grad_in", len(grad_in), [grad_in[i].shape if grad_in[i] is not None else None for i in range(len(grad_in))], grad_in, sep='\n')

        if self.accumulate_level == "sample":
            self.up_proj_grads[module.layer_index] += torch.sum(pass_kernel_function(grad_in[0].detach(), criterion=self.kernel), dim=0)
        elif self.accumulate_level == "total":
            self.up_proj_grads[module.layer_index] += torch.sum(grad_in[0].detach(), dim=0)
        else:
            raise NotImplementedError

        # if self.device == "cuda:0" and module.layer_index == 0:
        #     print(self.up_proj_grads[module.layer_index])

    def _backward_hook_gate_proj(self, module, grad_in, grad_out):
        if module.add_batch_size:
            batch_size = grad_out[0].shape[0]
            self.sample_count += batch_size - 1  # gradient of the last token in the batch is 0
            # print("sample_count:", self.sample_count)

        # if self.device == "cuda:0":
        #     with torch.cuda.device("cuda:0"):
        #         print("Used GPU memory (GPU 0): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")
        #     print("gate", "grad_out", len(grad_out), [grad_out[i].shape if grad_out[i] is not None else None for i in range(len(grad_out))], grad_out, sep='\n')
        #     print("gate", "grad_in", len(grad_in), [grad_in[i].shape if grad_in[i] is not None else None for i in range(len(grad_in))], grad_in, sep='\n')

        if self.accumulate_level == "sample":
            self.gate_proj_grads[module.layer_index] += torch.sum(pass_kernel_function(grad_out[0].detach(), criterion=self.kernel), dim=0)
        elif self.accumulate_level == "total":
            self.gate_proj_grads[module.layer_index] += torch.sum(grad_out[0].detach(), dim=0)
        else:
            raise NotImplementedError

        # if self.device == "cuda:0" and module.layer_index == 0:
        #     print(self.gate_proj_grads[module.layer_index])

    def get_grad(self):
        # initialization
        for layer_index, layer in enumerate(self.trainer.model.model.layers):  # locate block by the name template
            assert type(layer.mlp) == LlamaMLP
            if layer_index == 31:
                layer.mlp.down_proj.add_batch_size = True  # use the down_proj of layer0 to count for the batch_size
                layer.mlp.gate_proj.add_batch_size = False
            else:
                layer.mlp.down_proj.add_batch_size = False
                layer.mlp.gate_proj.add_batch_size = False

            layer.mlp.down_proj.layer_index = layer_index
            layer.mlp.down_proj.register_backward_hook(self._backward_hook_up_proj)  # "grad_out" of "down_proj" <==> grad of "up_proj * gate_proj" output
            layer.mlp.gate_proj.layer_index = layer_index
            layer.mlp.gate_proj.register_backward_hook(self._backward_hook_gate_proj)  # "grad_in" of "gate_proj" <==> grad of "gate_proj" output

        # get grads
        self.trainer.train()
        self.sample_count = torch.tensor(self.sample_count, device=self.device)

        # save grads
        # gather results on different devices
        gathered_sample_count = self.trainer.accelerator.gather(self.sample_count)
        gathered_sample_count = torch.sum(gathered_sample_count)

        # if self.device == "cuda:0":
        #     print(gathered_sample_count)

        if self.device == "cuda:0":
            if not os.path.exists(self.config.save_path):
                os.makedirs(self.config.save_path)

        for layer_index in tqdm(range(self.layer_num)):
            # gather results on different devices
            gathered_up_proj_grads = self.trainer.accelerator.gather(self.up_proj_grads[layer_index].reshape(1, -1))
            gathered_up_proj_grads = torch.sum(gathered_up_proj_grads, dim=0)
            gathered_gate_proj_grads = self.trainer.accelerator.gather(self.gate_proj_grads[layer_index].reshape(1, -1))
            gathered_gate_proj_grads = torch.sum(gathered_gate_proj_grads, dim=0)

            if self.device == "cuda:0":
                # accumulate at last if set to "total"
                if self.accumulate_level == "total":
                    gathered_up_proj_grads = pass_kernel_function(gathered_up_proj_grads, criterion=self.kernel)
                    gathered_gate_proj_grads = pass_kernel_function(gathered_gate_proj_grads, criterion=self.kernel)

                # get mean values
                gathered_up_proj_grads /= gathered_sample_count
                gathered_gate_proj_grads /= gathered_sample_count

                # print("gathered_up_proj_grads_mean")
                # print(gathered_up_proj_grads)

                # save
                up_filename = os.path.join(self.config.save_path, "layers.{}.mlp.up_proj.weight.grad".format(layer_index))
                torch.save(gathered_up_proj_grads.cpu(), up_filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Accumulated gradients of neurons for layer {layer_index} saved to "{up_filename}".')

                gate_filename = os.path.join(self.config.save_path, "layers.{}.mlp.gate_proj.weight.grad".format(layer_index))
                torch.save(gathered_gate_proj_grads.cpu(), gate_filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Accumulated gradients of neurons for layer {layer_index} saved to "{gate_filename}".')
    # fmt: on


class GradientSplit(LayerSplit):
    # fmt: off
    def __init__(self, config, template, layer, grad_list):
        super().__init__(config, template, layer)
        self.grad_list = grad_list
        self.num_experts = len(grad_list)
        self.neuron_num = grad_list[0].size(0)
        self.labels = np.zeros((self.neuron_num,))

    def sort_by_criterion(self, criterion):
        sorted_grad_list = []
        sorted_index_list = []

        for grads in self.grad_list:
            if criterion == "min":
                sorted_grads, sorted_indices = grads.sort(0)
            elif criterion == "max":
                sorted_grads, sorted_indices = grads.sort(0, descending=True)
            else:
                raise NotImplementedError

            sorted_grad_list.append(sorted_grads.tolist())
            sorted_index_list.append(sorted_indices.tolist())

        return sorted_grad_list, sorted_index_list

    def split_without_neuron_sharing(self, expert_size, criterion):
        sorted_grad_list, sorted_index_list = self.sort_by_criterion(criterion)

        # iterate over the "sorted_grad_list" and compare
        # O(neuron_num * num_experts) Complexity
        neuron_used_mark = [False] * self.neuron_num
        expert_neuron_count = [0] * self.num_experts
        expert_start_index = [0] * self.num_experts

        if criterion == "min":
            while sum(expert_neuron_count) < self.neuron_num:
                now_selected_grad = float('inf')
                now_selected_neuron = -1
                now_selected_expert = -1

                for expert_id in range(self.num_experts):
                    if expert_neuron_count[expert_id] == expert_size or expert_start_index[expert_id] == self.neuron_num:
                        continue

                    while expert_start_index[expert_id] < self.neuron_num:
                        if neuron_used_mark[sorted_index_list[expert_id][expert_start_index[expert_id]]]:
                            expert_start_index[expert_id] += 1
                        else:
                            break

                    if sorted_grad_list[expert_id][expert_start_index[expert_id]] <= now_selected_grad:  # ----- different here -----
                        now_selected_grad = sorted_grad_list[expert_id][expert_start_index[expert_id]]
                        now_selected_neuron = sorted_index_list[expert_id][expert_start_index[expert_id]]
                        now_selected_expert = expert_id

                self.labels[now_selected_neuron] = now_selected_expert
                assert (not neuron_used_mark[now_selected_neuron])
                neuron_used_mark[now_selected_neuron] = True
                expert_neuron_count[now_selected_expert] += 1
                expert_start_index[now_selected_expert] += 1

                # print(now_selected_neuron, now_selected_expert)

        elif criterion == "max":
            while sum(expert_neuron_count) < self.neuron_num:
                now_selected_grad = float('-inf')
                now_selected_neuron = -1
                now_selected_expert = -1

                for expert_id in range(self.num_experts):
                    if expert_neuron_count[expert_id] == expert_size or expert_start_index[expert_id] == self.neuron_num:
                        continue

                    while expert_start_index[expert_id] < self.neuron_num:
                        if neuron_used_mark[sorted_index_list[expert_id][expert_start_index[expert_id]]]:
                            expert_start_index[expert_id] += 1
                        else:
                            break

                    if sorted_grad_list[expert_id][expert_start_index[expert_id]] >= now_selected_grad:  # ----- different here -----
                        now_selected_grad = sorted_grad_list[expert_id][expert_start_index[expert_id]]
                        now_selected_neuron = sorted_index_list[expert_id][expert_start_index[expert_id]]
                        now_selected_expert = expert_id

                self.labels[now_selected_neuron] = now_selected_expert
                assert (not neuron_used_mark[now_selected_neuron])
                neuron_used_mark[now_selected_neuron] = True
                expert_neuron_count[now_selected_expert] += 1
                expert_start_index[now_selected_expert] += 1

                # print(now_selected_neuron, now_selected_expert)

        else:
            raise NotImplementedError

        # print(neuron_used_mark)
        # print(expert_neuron_count)
        # print(expert_start_index)

    def split_with_neuron_sharing(self, expert_size, criterion):
        sorted_grad_list, sorted_index_list = self.sort_by_criterion(criterion)
        self.labels = [sorted_index_list[i][:expert_size] for i in range(len(sorted_index_list))]  # 与其他的labels不同，此处选择的是神经元索引，而非专家索引

    def split(self, expert_size, criterion="min", share_neurons=False):
        if not share_neurons:
            assert expert_size * self.num_experts == self.neuron_num
            self.split_without_neuron_sharing(expert_size, criterion)
        else:
            self.split_with_neuron_sharing(expert_size, criterion)
    # fmt: on
