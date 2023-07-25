import os
from collections import Counter

import sklearn
import torch
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import MultiLabelBinarizer


def load_ffn_weight(model, template, layer):
    key = template.format(layer)
    return model[key].numpy()


class LayerSplit:
    def __init__(self, config, model, template, layer=0):
        self.config = config
        self.layer = layer
        self.template = template
        self.model = model
        self.model_dict = model.state_dict()

    def save(self):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        filename = os.path.join(self.config.save_path, self.template.format(self.layer))
        torch.save(self.labels, filename)
        print(f'Expert indices for layer {self.layer} saved to "{filename}".')

    def cnt(self):
        print(Counter(self.labels))

    def load_param(self):
        self.ffn_weight = load_ffn_weight(self.model_dict, self.template, self.layer)
        self.neuron_num = self.ffn_weight.shape[0]
        self.split_size = self.neuron_num // self.config.num_experts
        assert self.split_size * self.config.num_experts == self.neuron_num


class ClusteringSplit(LayerSplit):
    def __init__(self, config, model, template, layer=0):
        super().__init__(config, model, template, layer=layer)
        self.type = "split_clustering"

    def split(self):
        self.load_param()
        ffn_weight_norm = sklearn.preprocessing.normalize(self.ffn_weight)
        kmeans = KMeansConstrained(
            n_clusters=self.config.num_experts,
            size_min=self.split_size,
            size_max=self.split_size,
            max_iter=500,
            random_state=0,
            n_jobs=-1,
            verbose=True,
        ).fit(ffn_weight_norm, None)
        self.labels = [x for x in kmeans.labels_]
