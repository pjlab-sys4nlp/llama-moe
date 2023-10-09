import os
import pickle

import numpy as np
import torch

from smoe.utils.seed import set_seed


class LayerPrune:
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


class GradientPrune(LayerPrune):
    # fmt: off
    def __init__(self, config, template, layer, score):
        super().__init__(config, template, layer)
        self.score = score
        self.num_experts = 1
        self.neuron_num = score.size(0)

    def sort_by_criterion(self, criterion):
        if criterion == "min":
            sorted_score, sorted_index = self.score.sort(0)
        elif criterion == "max":
            sorted_score, sorted_index = self.score.sort(0, descending=True)
        else:
            raise NotImplementedError
        return sorted_score.tolist(), sorted_index.tolist()

    def prune(self, expert_size, criterion="min"):
        sorted_score, sorted_index = self.sort_by_criterion(criterion)
        self.labels = [sorted_index[:expert_size]]  # 与其他的labels不同，此处选择的是神经元索引，而非专家索引
        # print(self.labels)
        # fmt: on


class RandomPrune(LayerPrune):
    # fmt: off
    def __init__(self, config, template, layer, neuron_num):
        super().__init__(config, template, layer)
        self.num_experts = 1
        self.neuron_num = neuron_num

    def prune(self, expert_size, seed=None):
        if seed is not None:
            set_seed(seed)
        index = torch.randperm(self.neuron_num).tolist()
        self.labels = [index[:expert_size]]  # 与其他的labels不同，此处选择的是神经元索引，而非专家索引
        # print(self.labels)
    # fmt: on
