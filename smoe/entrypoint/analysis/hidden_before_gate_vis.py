import warnings
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from smoe.data.collate_fn import fault_tolerance_data_collator
from smoe.data.streaming import CachedJsonlDataset
from smoe.models.llama_moe import LlamaMoEForCausalLM
from smoe.models.llama_moe.modeling_llama_moe import (
    LlamaMoEDecoderLayer,
    MoEDecoderLayerOutput,
    MoEMlpOutput,
)
from smoe.modules.moe.moe_gates import TopKBalancedNoisyGate

eval_path_map = {
    "en_wikipedia": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/en_wikipedia.jsonl",
    "github": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/github.jsonl",
    "en_stack": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/en_stack.jsonl",
    "en_cc": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/en_cc.jsonl",
    "en_c4": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/en_c4.jsonl",
    "en_book": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/en_book.jsonl",
    "en_arxiv": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/en_arxiv.jsonl",
    "arc_challenge": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/arc_challenge.jsonl",
    "gsm8k": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/gsm8k.jsonl",
    "hellaswag": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/hellaswag.jsonl",
    "mmlu": "/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/mmlu.jsonl",
}
hidden_list = []


def hidden_recording_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
) -> MoEDecoderLayerOutput:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_list.append(hidden_states.detach().cpu())

    mlp_outs: MoEMlpOutput = self.mlp(hidden_states)
    hidden_states = residual + mlp_outs.hidden_states

    outputs = (
        hidden_states,
        mlp_outs.balance_loss,
        mlp_outs.num_dropped_tokens,
        mlp_outs.gate_load,
        mlp_outs.gate_importance,
    )
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)

    for i, _o in enumerate(outputs):
        if not isinstance(_o, torch.Tensor):
            raise RuntimeError(
                f"outputs[{i}]({type(_o)}) should be torch.Tensor to support grad ckpt"
            )

    return outputs


softmax_list = []


def gate_recording_forward(self, x):
    """先计算所有专家的权重值"""
    logits_gate = self.gate_network(x)  # gate计算出的权重
    logits = logits_gate  # 最终权重，shape(batch_size, num_experts)

    """选出前k个权重，并计算各个专家的分数scores"""
    top_logits, top_indices = logits.topk(
        min(self.num_selects + 1, self.num_experts), dim=1
    )  # 选择并排序前k+1个权重
    top_k_logits = top_logits[:, : self.num_selects]
    top_k_indices = top_indices[:, : self.num_selects]
    top_k_scores = self.softmax(top_k_logits)

    """计算importance"""
    zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
    scores_filtered = zeros.scatter(
        dim=1, index=top_k_indices, src=top_k_scores
    )  # shape(batch_size, num_experts)
    softmax_list.append(scores_filtered.detach().cpu())
    importance = scores_filtered.sum(0)  # shape(num_experts)

    """计算load"""
    load = (scores_filtered > 0).sum(0)

    """计算balance loss"""
    if self.use_balance:
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= self.balance_loss_weight
    else:
        balance_loss = torch.tensor(-100.0, device=x.device)

    return {
        "topK_indices": top_k_indices,
        "topK_scores": top_k_scores,
        "balance_loss": balance_loss,
        "load": load,
        "importance": importance,
    }


def plot_2d_distribution(embs, labels, save_path, title: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for emb, label in zip(embs, labels):
        ax.scatter(emb[:, 0], emb[:, 1], alpha=0.2, label=label, s=2)
    if title is not None:
        ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def tsne_for_one_layer(data_list, save_path, labels, title: str = None):
    # data_list: (num datasets, num tokens, hidden dim)
    tsne = TSNE(n_components=2, verbose=1)
    min_num = min([len(data) for data in data_list])

    embs = []
    for data in data_list:
        emb = tsne.fit_transform(data[:min_num])
        embs.append(emb)
    plot_2d_distribution(embs, labels, save_path, title=title)


def tsne_for_layers(data_list, save_dir, labels):
    num_layers = len(data_list[0])
    for layer_idx in trange(num_layers, desc="Making scatter plots"):
        tsne_for_one_layer(
            [d[layer_idx] for d in data_list],
            f"{save_dir}/tsne_L{layer_idx}.png",
            labels,
            title=f"t-SNE Layer {layer_idx}",
        )


def pca_for_one_layer(data_list, save_path, labels, title: str = None):
    # data_list: (num datasets, num tokens, hidden dim)
    pca = PCA(n_components=2)
    min_num = min([len(data) for data in data_list])

    embs = []
    for data in data_list:
        emb = pca.fit_transform(data[:min_num])
        embs.append(emb)
    plot_2d_distribution(embs, labels, save_path, title=title)


def pca_for_layers(data_list, save_dir, labels):
    num_layers = len(data_list[0])
    for layer_idx in trange(num_layers, "Making scatter plots"):
        pca_for_one_layer(
            [d[layer_idx] for d in data_list],
            f"{save_dir}/pca_L{layer_idx}.png",
            labels,
            title=f"PCA Layer {layer_idx}",
        )


@torch.no_grad()
def main(
    model_dir: str, result_dir: str, eval_datanames: list[str], load_cache: bool = False
):
    name2hidden = {}
    name2softmax = {}
    if load_cache:
        for name in eval_datanames:
            name2hidden[name] = np.load(f"{result_dir}/{name}_hidden.npy")
            name2softmax[name] = np.load(f"{result_dir}/{name}_softmax.npy")
    else:
        global hidden_list
        global softmax_list
        bsz = 8
        result_dir = Path(result_dir)
        result_dir.mkdir(exist_ok=True, parents=True)

        accel = Accelerator()
        model = LlamaMoEForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        for module in model.modules():
            if isinstance(module, LlamaMoEDecoderLayer):
                module.forward = MethodType(hidden_recording_forward, module)
            if isinstance(module, TopKBalancedNoisyGate):
                module.forward = MethodType(gate_recording_forward, module)

        model.eval()
        model = accel.prepare_model(model)

        eval_dataset = {
            name: CachedJsonlDataset(eval_path_map[name], seed=1227, block_size=4096)
            for name in eval_datanames
        }
        for name, eval_dataset in eval_dataset.items():
            hidden_list = []
            tot_hidden = []
            softmax_list = []
            tot_softmax = []
            loader = DataLoader(
                eval_dataset, batch_size=bsz, collate_fn=fault_tolerance_data_collator
            )
            loader = accel.prepare_data_loader(loader)
            if name == "en_book":
                num_batch = 20
            else:
                num_batch = 9999999999999999
            for batch_idx, batch in enumerate(tqdm(loader, desc=name)):
                if batch_idx >= num_batch:
                    break
                model(**batch, output_attentions=False, use_cache=False)
                _tmp_batch_hidden = torch.stack(hidden_list, dim=0)
                # (num layers, num tokens, hidden dim)
                _tmp_batch_hidden = _tmp_batch_hidden.reshape(
                    len(hidden_list), -1, _tmp_batch_hidden.shape[-1]
                )
                tot_hidden.append(_tmp_batch_hidden.detach().cpu().float().numpy())
                _tmp_batch_softmax = torch.stack(softmax_list, dim=0)
                # (num layers, num tokens, num experts)
                _tmp_batch_softmax = _tmp_batch_softmax.reshape(
                    len(softmax_list), -1, _tmp_batch_softmax.shape[-1]
                )
                tot_softmax.append(_tmp_batch_softmax.detach().cpu().float().numpy())
                hidden_list = []
                softmax_list = []
            # (num layers, num tokens across all batches, hidden dim)
            tot_hidden = np.concatenate(tot_hidden, axis=1)
            # (num layers, num tokens across all batches, expert num)
            tot_softmax = np.concatenate(tot_softmax, axis=1)
            np.save(result_dir / f"{name}_hidden.npy", tot_hidden)
            np.save(result_dir / f"{name}_softmax.npy", tot_softmax)
            name2hidden[name] = tot_hidden
            name2softmax[name] = tot_softmax

    # data_list = []
    # labels = []
    # for name, hidden in name2hidden.items():
    #     data_list.append(hidden)
    #     labels.append(name)
    # tsne_for_layers(data_list, result_dir, labels=labels)
    # pca_for_layers(data_list, result_dir, labels=labels)


def heatmap(arr: np.ndarray, save_path: str, title: str, vmin: float, vmax: float):
    shape = arr.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, cmap="OrRd", interpolation="nearest", vmin=vmin, vmax=vmax)
    for row in range(shape[0]):
        for col in range(shape[1]):
            ax.text(
                col,
                row,
                f"{arr[row, col]:.4f}",
                ha="center",
                va="center",
                color="black",
            )
    ax.set_axis_off()
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path)


def dual_heatmap(arr1, arr2, save_path, layer_idx: int):
    shape = arr1.shape
    assert arr1.shape == arr2.shape
    vmin = min(arr1.min(), arr2.min())
    vmax = max(arr2.max(), arr2.max())
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(arr1, cmap="OrRd", interpolation="nearest", vmin=vmin, vmax=vmax)
    for row in range(shape[0]):
        for col in range(shape[1]):
            ax1.text(
                col,
                row,
                f"{arr1[row, col]:.4f}",
                ha="center",
                va="center",
                color="black",
            )
    ax1.set_title("GSM8K")
    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(arr2, cmap="OrRd", interpolation="nearest", vmin=vmin, vmax=vmax)
    for row in range(shape[0]):
        for col in range(shape[1]):
            ax2.text(
                col,
                row,
                f"{arr2[row, col]:.4f}",
                ha="center",
                va="center",
                color="black",
            )
    ax2.set_title("MMLU")
    ax1.set_axis_off()
    ax2.set_axis_off()
    fig.suptitle(f"Mean Routing Prob Layer {layer_idx}")
    fig.tight_layout()

    fig.savefig(save_path)


def dual_hist(
    arr1, arr2, save_path, layer_idx: int, xlim: list = None, ylim: list = None
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(arr1.flatten(), bins=100, label="GSM8K", alpha=0.5)
    ax.hist(arr2.flatten(), bins=100, label="MMLU", alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.set_title(f"Mean Routing Prob Layer {layer_idx}")
    fig.tight_layout()

    fig.savefig(save_path)


def softmax_vis(name, cache_filepath, save_dir, vmin, vmax):
    # (num layers, num tokens across all batches, expert num)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    vals = np.load(cache_filepath)
    for layer_idx, layer_vals in enumerate(vals):
        val = layer_vals.mean(axis=0)
        val = val.reshape(4, 4)
        heatmap(
            val,
            f"{save_dir}/softmax_L{layer_idx}.png",
            f"{name} Routing Mean Prob Layer {layer_idx}",
            vmin,
            vmax,
        )


if __name__ == "__main__":
    # w/ fluency filtering, 90b
    model_dir = "/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_112gpus_dynamic_data/outputs/cpt-llama2_random_scale4_112gpus_dynamic_data-2326233/checkpoint-6120"
    result_dir = "/mnt/petrelfs/zhutong/smoe/results/llama2_7B_random_split_sheared_sampling_fluency_90B_hidden_dist_vis"

    Path(result_dir).mkdir(exist_ok=True, parents=True)

    # ---- hidden hist ----
    Path(result_dir).joinpath("hidden_hist").mkdir(exist_ok=True, parents=True)
    vals1 = np.load(f"{result_dir}/gsm8k_hidden.npy")
    vals2 = np.load(f"{result_dir}/mmlu_hidden.npy")
    for layer_idx, (layer_vals1, layer_vals2) in enumerate(zip(vals1, vals2)):
        num_tokens1 = layer_vals1.shape[0]
        num_tokens2 = layer_vals2.shape[0]
        limit = min(num_tokens1, num_tokens2)
        layer_vals1 = layer_vals1[:limit]
        layer_vals2 = layer_vals2[:limit]
        assert layer_vals1.shape == layer_vals2.shape

        dual_hist(
            layer_vals1,
            layer_vals2,
            f"{result_dir}/hidden_hist/dual_hidden_L{layer_idx}.png",
            layer_idx,
            xlim=(-2, 2),
        )

    # # ---- gate softmax values ----
    # Path(result_dir).joinpath("dual").mkdir(exist_ok=True, parents=True)
    # Path(result_dir).joinpath("dual_hist").mkdir(exist_ok=True, parents=True)
    # vals1 = np.load(f"{result_dir}/gsm8k_softmax.npy")
    # vals2 = np.load(f"{result_dir}/mmlu_softmax.npy")
    # for layer_idx, (layer_vals1, layer_vals2) in enumerate(zip(vals1, vals2)):
    #     num_tokens1 = layer_vals1.shape[0]
    #     num_tokens2 = layer_vals2.shape[0]
    #     limit = min(num_tokens1, num_tokens2)
    #     layer_vals1 = layer_vals1[:limit]
    #     layer_vals2 = layer_vals2[:limit]
    #     assert layer_vals1.shape == layer_vals2.shape

    #     val1 = layer_vals1.mean(axis=0)
    #     val2 = layer_vals2.mean(axis=0)
    #     val1 = val1.reshape(4, 4)
    #     val2 = val2.reshape(4, 4)
    #     dual_heatmap(
    #         val1,
    #         val2,
    #         f"{result_dir}/dual/dual_softmax_L{layer_idx}.png",
    #         layer_idx,
    #     )

    #     val1_ind = np.argsort(layer_vals1, axis=1)
    #     top4_ind = val1_ind[:, -4:]
    #     val1 = np.take_along_axis(layer_vals1, top4_ind, axis=1)
    #     val2_ind = np.argsort(layer_vals2, axis=1)
    #     top4_ind = val2_ind[:, -4:]
    #     val2 = np.take_along_axis(layer_vals2, top4_ind, axis=1)
    #     dual_hist(
    #         val1,
    #         val2,
    #         f"{result_dir}/dual_hist/dual_softmax_L{layer_idx}.png",
    #         layer_idx,
    #     )
