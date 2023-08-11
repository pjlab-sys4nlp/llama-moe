import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from transformers import LlamaModel

from smoe.modules.moe.moe_gates import TopKBalancedNoisyGate
from smoe.utils.kernel_function import pass_kernel_function


class BaseGate:
    def __init__(
        self,
        config,
        llama_model,
        train_loader,
        valid_loader,
        expert_indices,
        layer_index,
    ):
        assert type(llama_model) == LlamaModel

        self.config = config
        self.llama_model = llama_model
        self.llama_model_dict = llama_model.state_dict()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.expert_indices = expert_indices
        self.layer_index = layer_index

        self.hidden_dim = self.llama_model.config.hidden_size

        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)


class MLPGate(BaseGate):
    def __init__(
        self,
        config,
        llama_model,
        train_loader,
        valid_loader,
        expert_indices,
        layer_index,
        select_criterion="plain",
        criterion_config=None,
    ):
        super().__init__(
            config, llama_model, train_loader, valid_loader, expert_indices, layer_index
        )
        self.available_criterion = ("plain", "positive", "l2_norm")
        assert select_criterion in self.available_criterion

        self.type = "select_mlp"

        self.select_criterion = select_criterion
        self.criterion_config = criterion_config

    def mlp_gate_weights_init(self, module):
        # fmt: off
        if isinstance(module, torch.nn.Linear):
            if module.weight.shape[-1] == self.hidden_dim:  # 第一层，用所有专家权重的中心点初始化，shape(expert_num, input_dim)
                module.weight.data = torch.from_numpy(self.centers).float()  # 不理解有什么道理，这样会使gate初始偏向选择与权重方向最相似的输入，使其缺乏变换能力
            else:  # 第二层，使用对角矩阵初始化，意图平衡专家间的知识
                module.weight.data = torch.eye(module.weight.data.shape[0])
                # torch.nn.init.normal_(m.weight.data)
            # m.bias.data[:] = 0
        # fmt: on

    def calculate_scores(self, hidden_outputs, expert_masks, use_softmax=False):
        # fmt: off
        with torch.no_grad():
            if self.select_criterion == "plain":
                scores = torch.matmul(hidden_outputs, expert_masks)  # 各个专家所对应神经元的输出总值，shape(batch_size, expert_num)
                # scores /= torch.abs(scores).max()  # 归一化

            elif self.select_criterion == "positive":
                hidden_outputs_mask = (hidden_outputs < 0)  # 选出输出值小于0的神经元，标记其为死神经元
                hidden_outputs[hidden_outputs_mask] = 0  # 死神经元的输出置零

                scores = torch.matmul(hidden_outputs, expert_masks)  # 各个专家所对应神经元的正向激活程度总值，shape(batch_size, expert_num)
                # scores /= scores.max()  # 归一化

            elif self.select_criterion == "l2_norm":
                threshold = 0.001 if self.criterion_config is None else self.criterion_config["threshold"]

                hidden_outputs_l2 = pass_kernel_function(hidden_outputs, criterion="l2_norm")  # 输出值L2范数
                hidden_outputs_mask = (hidden_outputs_l2 <= threshold)  # 选出输出值L2范数小于等于给定阈值的神经元，标记其为死神经元
                hidden_outputs_l2[hidden_outputs_mask] = 0  # 死神经元的输出置零

                scores = torch.matmul(hidden_outputs_l2, expert_masks)  # 各个专家所对应神经元的输出值L2范数总值，shape(batch_size, expert_num)
                # scores /= scores.max()  # 归一化

            if use_softmax:
                scores = nn.Softmax(1)(scores)

        return scores
        # fmt: on

    def train(
        self,
        device,
        batch_size=1024,
        train_epochs=100,
        lr=0.01,
        accumulate_steps=1,
        use_balance=False,
        add_noise=False,
        use_softmax=False,
        balance_loss_lambda=0.0005,
    ):
        """
        每轮epoch训练一层
        需要配合ShardDatasetForMoEGate作为dataloader使用
        """

        # fmt: off
        """create expert masks"""
        expert_masks = []
        for j in range(self.config.num_experts):
            expert_masks.append(np.array(self.expert_indices) == j)
        expert_masks = np.stack(expert_masks, axis=0)
        expert_masks = torch.tensor(expert_masks).float().transpose(0, 1).to(device)  # 各个专家所对应的神经元mask

        """Prepare models and optimizers"""
        # weights for initialization
        ffn_weight = self.llama_model_dict[self.config.template.format(self.layer_index)].numpy()  # llama的参数权重，shape(hidden_neurons, input_dim)
        ffn_weight_norm_ = Normalizer().transform(ffn_weight)

        centers = []
        for j in range(self.config.num_experts):
            centers.append(ffn_weight_norm_[np.array(self.expert_indices) == j, :].mean(0))  # shape(1, input_dim)
        self.centers = np.array(centers)  # 各个专家神经元所对应权重的均值中心，shape(expert_num, input_dim)

        # create models
        self.mlp_model = TopKBalancedNoisyGate(self.hidden_dim, self.config.num_experts, self.config.num_selects, gate_network="mlp",
                                               use_balance=use_balance, add_noise=add_noise, use_softmax=use_softmax)
        self.mlp_model.gate_network.apply(self.mlp_gate_weights_init)
        self.mlp_model = self.mlp_model.to(device)

        optimizer = torch.optim.AdamW(self.mlp_model.parameters(), lr=lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.85, threshold=0.03, patience=5)
        # loss_function = torch.nn.BCEWithLogitsLoss()
        loss_function = torch.nn.KLDivLoss(reduction="batchmean")

        """Initialize training configs"""
        self.train_loss = {
            "loss": [],
            "epochs": []
        }
        self.train_acc = {
            "acc": [],
            "epochs": []
        }
        self.valid_loss = {
            "loss": [],
            "epochs": []
        }
        self.valid_acc = {
            "acc": [],
            "epochs": []
        }
        self.save_loss = 1000000
        self.save_acc = 0
        self.save_epoch = -1

        """Start training"""
        with torch.cuda.device(device):
            print("Used GPU memory (device): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

        process_bar = tqdm(range(train_epochs), desc="epoch", leave=True)
        for epoch in process_bar:
            if epoch >= train_epochs:
                break
            train_loss_this_epoch = []
            train_acc_this_epoch = []
            valid_loss_this_epoch = []
            valid_acc_this_epoch = []

            """Training"""
            train_batch_cnt = 0
            iter_train = iter(self.train_loader)
            process_bar2 = tqdm(range(len(self.train_loader)), desc="training step", leave=False)
            for train_step in process_bar2:
                if train_step >= len(self.train_loader):
                    break
                train_loss_this_step = []
                train_acc_this_step = []
                hidden_inputs_examples, hidden_outputs_examples = next(iter_train)
                shuffle_index = torch.randperm(hidden_inputs_examples.shape[0])  # 随机打乱序号
                hidden_inputs_examples = torch.split(hidden_inputs_examples[shuffle_index].float().to(device), batch_size, dim=0)  # LLaMA模型MLP的输入
                hidden_outputs_examples = torch.split(hidden_outputs_examples[shuffle_index].float().to(device), batch_size, dim=0)  # LLaMA模型MLP的GLU门控输出

                # Forward mlp gates with hidden_states, compute loss and backward
                for batch_id in range(len(hidden_inputs_examples)):
                    train_batch_cnt += 1
                    hidden_inputs = hidden_inputs_examples[batch_id]
                    hidden_outputs = hidden_outputs_examples[batch_id]

                    pred, gate_loss = self.mlp_model.forward_return_scores(hidden_inputs, gate_loss_lambda=balance_loss_lambda)  # MoE gate选择的各个专家的scores，shape(batch_size, expert_num)
                    pred_topk, pred_labels = torch.topk(pred, k=int(self.config.num_selects), dim=-1)

                    scores = self.calculate_scores(hidden_outputs, expert_masks, use_softmax=use_softmax)  # 根据当前层激活后的输出所计算出的各个专家的分数
                    scores_topk, scores_labels = torch.topk(scores, k=int(self.config.num_selects), dim=-1)

                    # scores_reform = torch.zeros((batch_size, self.config.num_experts), device=device)
                    # scores_reform = scores_reform.scatter(dim=1, index=scores_labels, value=1)  # shape(batch_size, expert_num)

                    loss = loss_function(torch.log(pred.view(-1)), scores.view(-1))  # KL损失
                    loss += gate_loss
                    loss.backward()

                    correct_num = torch.sum(torch.eq(pred_labels, scores_labels)).item()
                    total_num = torch.numel(pred_labels)
                    acc = correct_num / total_num  # 计算预测正确的acc

                    if train_batch_cnt % accumulate_steps == (accumulate_steps - 1):
                        optimizer.step()
                        self.mlp_model.zero_grad()

                    train_loss_this_step.append(loss.item())
                    train_acc_this_step.append(acc)
                train_loss_this_epoch.append(sum(train_loss_this_step) / len(train_loss_this_step))
                train_acc_this_epoch.append(sum(train_acc_this_step) / len(train_acc_this_step))

                self.train_loss["epochs"].append(epoch)
                self.train_loss["loss"].append(train_loss_this_epoch[-1])
                self.train_acc["epochs"].append(epoch)
                self.train_acc["acc"].append(train_acc_this_epoch[-1])

                process_bar2.set_postfix(avg_loss=train_loss_this_epoch[-1],
                                         avg_acc=train_acc_this_epoch[-1])
                process_bar2.update(1)
            process_bar2.close()

            """Validation"""
            iter_valid = iter(self.valid_loader)
            process_bar3 = tqdm(range(len(self.valid_loader)), desc="validation step", leave=False)
            with torch.no_grad():
                for valid_step in process_bar3:
                    if valid_step >= len(self.valid_loader):
                        break
                    valid_loss_this_step = []
                    valid_acc_this_step = []
                    hidden_inputs_examples, hidden_outputs_examples = next(iter_valid)
                    hidden_inputs_examples = torch.split(hidden_inputs_examples.float().to(device), batch_size, dim=0)  # LLaMA模型MLP的输入
                    hidden_outputs_examples = torch.split(hidden_outputs_examples.float().to(device), batch_size, dim=0)  # LLaMA模型MLP的GLU门控输出

                    # Forward mlp gates with hidden_states, compute loss and backward
                    for batch_id in range(len(hidden_inputs_examples)):
                        hidden_inputs = hidden_inputs_examples[batch_id]
                        hidden_outputs = hidden_outputs_examples[batch_id]

                        pred, gate_loss = self.mlp_model.forward_return_scores(hidden_inputs, gate_loss_lambda=balance_loss_lambda)  # MoE gate选择的各个专家的scores，shape(batch, expert_num)
                        pred_topk, pred_labels = torch.topk(pred, k=int(self.config.num_selects), dim=-1)

                        scores = self.calculate_scores(hidden_outputs, expert_masks, use_softmax=use_softmax)  # 根据当前层激活后的输出所计算出的各个专家的分数
                        scores_topk, scores_labels = torch.topk(scores, k=int(self.config.num_selects), dim=-1)

                        loss = loss_function(torch.log(pred.view(-1)), scores.view(-1))  # KL损失

                        correct_num = torch.sum(torch.eq(pred_labels, scores_labels)).item()
                        total_num = torch.numel(pred_labels)
                        acc = correct_num / total_num  # 计算预测正确的acc

                        valid_loss_this_step.append(loss.item())
                        valid_acc_this_step.append(acc)
                    valid_loss_this_epoch.append(sum(valid_loss_this_step) / len(valid_loss_this_step))
                    valid_acc_this_epoch.append(sum(valid_acc_this_step) / len(valid_acc_this_step))

                    self.valid_loss["epochs"].append(epoch)
                    self.valid_loss["loss"].append(valid_loss_this_epoch[-1])
                    self.valid_acc["epochs"].append(epoch)
                    self.valid_acc["acc"].append(valid_acc_this_epoch[-1])

                    process_bar3.set_postfix(avg_loss=valid_loss_this_epoch[-1],
                                             avg_acc=valid_acc_this_epoch[-1])
                    process_bar3.update(1)
            process_bar3.close()

            # lr_scheduler.step()
            lr_scheduler.step(train_loss_this_epoch[-1])

            """Save best models"""
            cur_acc = sum(valid_acc_this_epoch) / len(valid_acc_this_epoch)
            cur_loss = sum(valid_loss_this_epoch) / len(valid_loss_this_epoch)
            if cur_acc > self.save_acc:
                self.save_acc = cur_acc
                self.save_loss = cur_loss
                self.save_epoch = epoch
                self.save()
                self.mlp_model = self.mlp_model.to(device)

            process_bar.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'],
                                    train_loss=sum(train_loss_this_epoch) / len(train_loss_this_epoch),
                                    train_acc=sum(train_acc_this_epoch) / len(train_acc_this_epoch),
                                    valid_loss=sum(valid_loss_this_epoch) / len(valid_loss_this_epoch),
                                    valid_acc=sum(valid_acc_this_epoch) / len(valid_acc_this_epoch))
            process_bar.update(1)
        process_bar.close()

        """Save training statistics"""
        # Save loss and acc
        save_file_name = os.path.join(self.config.save_path, self.config.template.format(self.layer_index))
        with open("{}.log".format(save_file_name), 'w') as fout:
            fout.write("train_epochs: " + str(self.train_loss["epochs"]) + "\n")
            fout.write("train_loss: " + str([format(loss, '.4f') for loss in self.train_loss["loss"]]) + "\n")
            fout.write("train_acc: " + str([format(loss, '.4f') for loss in self.train_acc["acc"]]) + "\n")
            fout.write("valid_epochs: " + str(self.valid_loss["epochs"]) + "\n")
            fout.write("valid_loss: " + str([format(loss, '.4f') for loss in self.valid_loss["loss"]]) + "\n")
            fout.write("valid_acc: " + str([format(loss, '.4f') for loss in self.valid_acc["acc"]]) + "\n")
        # fmt: on

    def save(self):
        # fmt: off
        save_file_name = os.path.join(self.config.save_path, self.config.template.format(self.layer_index))

        # Save model
        print("\nSaving model \"mlp_layer_" + str(self.layer_index) + "\" to \"" + save_file_name + "\"...")
        torch.save(self.mlp_model.cpu().state_dict(), save_file_name)

        # Save acc
        with open("{}.acc".format(save_file_name), 'w') as fout:
            fout.write("best_acc: " + str(self.save_acc) + "\n")
            fout.write("best_loss: " + str(self.save_loss) + "\n")
            fout.write("save_epoch: " + str(self.save_epoch) + "\n")
        # fmt: on
