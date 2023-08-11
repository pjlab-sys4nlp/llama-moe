import os
import random
from collections import Counter

import numpy as np
import sklearn
import torch
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer  # noqa: F401

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
        torch.save(self.labels, filename)
        print(f'Expert indices for layer {self.layer} saved to "{filename}".')

    def cnt(self):
        print(Counter(self.labels))


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
        ffn_weight_norm = sklearn.preprocessing.normalize(self.ffn_weight)
        # ffn_weight_norm = Normalizer().transform(self.ffn_weight)

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
