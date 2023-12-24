import itertools

import numpy as np
import torch

from smoe.utils.expert_construction.expert_split import LayerSplit
from smoe.utils.list_operation import chunk_list
from smoe.utils.visualization.visualize import visualize_expert_neuron_overlap


class GradientSplitResidual(LayerSplit):
    # fmt: off
    def __init__(self, config, template, layer, score_list):
        super().__init__(config, template, layer)
        self.score_list = score_list
        self.neuron_num = score_list[0].size(0)

    def sort_by_criterion(self, criterion):
        sorted_score_list = []
        sorted_index_list = []

        for scores in self.score_list:
            if criterion == "min":
                sorted_scores, sorted_indices = scores.sort(0)
            elif criterion == "max":
                sorted_scores, sorted_indices = scores.sort(0, descending=True)
            else:
                raise NotImplementedError

            sorted_score_list.append(sorted_scores.tolist())
            sorted_index_list.append(sorted_indices.tolist())

        return sorted_score_list, sorted_index_list

    def remove_residual_neurons(self, sorted_index_list, residual_neuron_mask):
        new_residual_neuron_mask = []

        for indices in sorted_index_list:
            new_indices = []
            for index in indices:
                if not residual_neuron_mask[index]:
                    new_indices.append(index)
            new_residual_neuron_mask.append(new_indices)

        return new_residual_neuron_mask

    def split_without_neuron_sharing(self, expert_num_moe, expert_num_residual, expert_size, criterion):
        neuron_used_mark = [False] * self.neuron_num
        expert_start_index = [0] * expert_num_moe
        expert_neuron_count = [0] * expert_num_moe
        expert_neuron_count_total = 0

        # select residual neurons with the sharing algorithm
        self.split_with_neuron_sharing(expert_num_moe, expert_num_residual, expert_size, criterion)

        # clear labels for moe neurons
        for expert_id in range(expert_num_residual, expert_num_residual + expert_num_moe):
            self.labels[expert_id] = []

        # exclude neurons selected by the residual block
        residual_labels = self.labels[:expert_num_residual]
        for index in list(itertools.chain(*residual_labels)):
            neuron_used_mark[index] = True

        sorted_score_list, sorted_index_list = self.sort_by_criterion(criterion)

        # iterate over the "sorted_score_list" and compare
        # greedily select the maximum score from the highest score of each expert
        # O(neuron_num * expert_num_moe) time complexity
        moe_neuron_num = expert_num_moe * expert_size
        while expert_neuron_count_total < moe_neuron_num:
            if criterion == "min":
                now_selected_score = float('inf')
            elif criterion == "max":
                now_selected_score = float('-inf')
            else:
                raise NotImplementedError

            now_selected_neuron = -1
            now_selected_expert = -1

            for expert_id in range(expert_num_moe):
                while expert_start_index[expert_id] < self.neuron_num:
                    if neuron_used_mark[sorted_index_list[expert_id][expert_start_index[expert_id]]]:
                        expert_start_index[expert_id] += 1
                    else:
                        break

                if expert_neuron_count[expert_id] == expert_size or expert_start_index[expert_id] == self.neuron_num:
                    continue

                if criterion == "min":
                    if sorted_score_list[expert_id][expert_start_index[expert_id]] <= now_selected_score:  # ----- different here -----
                        now_selected_score = sorted_score_list[expert_id][expert_start_index[expert_id]]
                        now_selected_neuron = sorted_index_list[expert_id][expert_start_index[expert_id]]
                        now_selected_expert = expert_id
                elif criterion == "max":
                    if sorted_score_list[expert_id][expert_start_index[expert_id]] >= now_selected_score:  # ----- different here -----
                        now_selected_score = sorted_score_list[expert_id][expert_start_index[expert_id]]
                        now_selected_neuron = sorted_index_list[expert_id][expert_start_index[expert_id]]
                        now_selected_expert = expert_id
                else:
                    raise NotImplementedError

            self.labels[expert_num_residual + now_selected_expert].append(now_selected_neuron)
            assert (not neuron_used_mark[now_selected_neuron])
            neuron_used_mark[now_selected_neuron] = True
            expert_start_index[now_selected_expert] += 1
            expert_neuron_count[now_selected_expert] += 1
            expert_neuron_count_total += 1
            # print(now_selected_expert, now_selected_neuron)
            # print(expert_neuron_count)
            # print(expert_start_index)

        # print(neuron_used_mark)
        # print(expert_neuron_count)
        # print(expert_start_index)

    def split_with_neuron_sharing(self, expert_num_moe, expert_num_residual, expert_size, criterion="min"):
        sorted_score_list, sorted_index_list = self.sort_by_criterion(criterion)

        # 与其他的labels不同，此处选择的是神经元索引，而非专家索引
        residual_labels = []  # residual各专家选择的神经元索引(共享神经元)
        residual_neuron_mask = [False] * self.neuron_num  # 标识哪些神经元是共享的

        # iteratively assign the indices of mostly shared neurons to the residual block
        while not len(residual_labels) >= expert_num_residual * expert_size:
            moe_labels = [sorted_index_list[i][:expert_size] for i in range(expert_num_moe)]  # moe各专家选择的神经元索引

            selected_count = torch.zeros((self.neuron_num,), dtype=torch.int)
            for selected_indices in moe_labels:
                selected_indices = torch.tensor(selected_indices)
                selected_count[selected_indices] += 1

            for repeat_times in range(expert_num_moe, 0, -1):
                repeat_mask = (selected_count == repeat_times)
                selected_neurons_count = torch.sum(repeat_mask).item()
                if selected_neurons_count > 0:
                    residual_indices = torch.nonzero(repeat_mask).flatten().tolist()
                    if len(residual_indices) > expert_num_residual * expert_size - len(residual_labels):  # 如果添加后会超过residual容量上限
                        residual_indices = residual_indices[:expert_num_residual * expert_size - len(residual_labels)]

                    # print(residual_indices)
                    residual_labels.extend(residual_indices)  # 添加到residual_labels
                    for index in residual_indices:  # 更新标识
                        residual_neuron_mask[index] = True
                    print(f"Selected {selected_neurons_count} from repeat {repeat_times}. Total {len(residual_labels)}")
                    break

            sorted_index_list = self.remove_residual_neurons(sorted_index_list, residual_neuron_mask)

        print(f"Final {len(residual_labels)} residual.")
        print(f"Final {self.neuron_num - len(residual_labels)} moe.")

        residual_labels = chunk_list(residual_labels, expert_num_residual)  # residual各专家选择的神经元索引(共享神经元)
        moe_labels = [sorted_index_list[i][:expert_size] for i in range(expert_num_moe)]  # moe各专家选择的神经元索引

        self.labels = residual_labels
        self.labels.extend(moe_labels)

    def split(self, expert_num_moe, expert_num_residual, expert_size, criterion="min", share_neurons=False):
        assert expert_size <= self.neuron_num
        if not share_neurons:
            # print("***", expert_size, expert_num, self.neuron_num)
            assert expert_size * (expert_num_moe + expert_num_residual) == self.neuron_num
            self.split_without_neuron_sharing(expert_num_moe, expert_num_residual, expert_size, criterion)
        else:
            self.split_with_neuron_sharing(expert_num_moe, expert_num_residual, expert_size, criterion)

    def visualize(self, save_path, share_neurons=False):
        if share_neurons:  # 必须在share_neuron的情况下才可以可视化
            num_experts = len(self.labels)
            expert_size = len(self.labels[0])

            selected_mask_list = []
            for i, indices in enumerate(self.labels):
                indices_tensor = torch.tensor(indices)
                selected_mask = torch.zeros((self.neuron_num,), dtype=torch.int)
                selected_mask[indices_tensor] += 1
                selected_mask_list.append(selected_mask)
            selected_masks = torch.stack(selected_mask_list, dim=0)  # shape(num_experts, intermediate_size)

            visualize_expert_neuron_overlap(selected_masks, num_experts, self.neuron_num, expert_size, self.layer, save_dir=save_path)
        else:
            print("Skip visualization as share_neurons==False.")
    # fmt: on
