import argparse
import os
import types

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

from smoe.data.collate_fn import tensor_dict_cat_collator
from smoe.data.datasets_moefication import LineByLineJsonlTextDataset
from smoe.models.llama_moefication import LlamaMoEForCausalLM
from smoe.modules.moe.moe_gates import TopKBalancedNoisyGate
from smoe.utils.change_llama_moe_forward import (
    forward_linear_glu_moe_layer_with_padding_mask,
    forward_llama_moe_decoder_with_padding_mask,
    forward_llama_moe_model_with_padding_mask,
    forward_mlp_moe_gate_with_load_recording,
)
from smoe.utils.visualization.visualize import (
    visualize_expert_load_barv,
    visualize_expert_load_heatmap,
)


# fmt: off
def change_forward(llama_model):
    llama_model.forward = types.MethodType(forward_llama_moe_model_with_padding_mask, llama_model)  # change forward function for LlamaModel

    for layer_idx, layer in enumerate(llama_model.layers):  # locate block by the name template
        assert type(layer.mlp.gate) == TopKBalancedNoisyGate

        layer.forward = types.MethodType(forward_llama_moe_decoder_with_padding_mask, layer)  # change forward function for LlamaDecoderLayer
        layer.mlp.forward = types.MethodType(forward_linear_glu_moe_layer_with_padding_mask, layer.mlp)  # change forward function for LinearGLUMoELayer
        layer.mlp.gate.forward = types.MethodType(forward_mlp_moe_gate_with_load_recording, layer.mlp.gate)  # change forward function TopKBalancedNoisyGate

        layer.mlp.gate.load_sum = torch.zeros((llama_model.config.num_experts,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/llama_7B")
    parser.add_argument('--model_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_model_bak/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-23000/commoncrawl-part-000203-16de0c55-head1000.jsonl")
    parser.add_argument('--data_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/data/vis_data")
    parser.add_argument('--save_path', type=str, default="/mnt/petrelfs/share_data/quxiaoye/visualization")
    parser.add_argument('--batch_size', type=int, default=8)  # 单次evaluate的batch_size

    args = parser.parse_args()
    print("\n", args)

    data_index_range = (0, 200) ##########################

    print("\ncuda is_available: " + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """load tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    """prepare datasets"""
    print("\nReading dataset from file \"" + args.data_path + "\"...")
    dataset = LineByLineJsonlTextDataset(tokenizer, file_path=args.data_path, block_size=2048, data_index_range=data_index_range)
    print(f"Dataset: {sum([torch.sum(dataset[i]['attention_mask']).item() for i in range(len(dataset))])} total tokens.")  # 统计非special token的数量

    """prepare dataloader"""
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=tensor_dict_cat_collator, num_workers=8, pin_memory=True, persistent_workers=True)

    """load model"""
    print("Loading llama model...")
    model = LlamaMoEForCausalLM.from_pretrained(args.model_path).model
    change_forward(model)

    """evaluation"""
    print("Start evaluation...")
    model.to(device)
    model.eval()
    iter_train = iter(data_loader)
    for step in tqdm(range(len(data_loader)), desc="forward step", position=0, leave=True):
        batch = next(iter_train)
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model(**batch)

    """visualization"""
    dataset_name = os.path.split(args.data_path)[1].split(".")[0]
    for layer_idx, layer in enumerate(model.layers):
        load_sum = layer.mlp.gate.load_sum.cpu().numpy()  # shape(num_experts)
        visualize_expert_load_heatmap(
            load_sum,
            layer_idx,
            dataset_name,
            save_dir=args.save_path + "-heat"
        )
        visualize_expert_load_barv(
            load_sum,
            layer_idx,
            dataset_name,
            save_dir=args.save_path + "-bar"
        )
    print("Done.")
