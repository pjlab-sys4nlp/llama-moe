import argparse
import os

from smoe.models.llama_moe.modeling_llama_moe import (
    LlamaMoEForCausalLM,
    LlamaMoEForSequenceClassification,
    LlamaMoEModel,
)
from smoe.utils.moefication.convert_llama_moe_neuron_index import (
    convert_llama_model_for_causal_lm_neuron_index,
    convert_llama_model_for_sequence_classification_neuron_index,
    convert_llama_model_neuron_index,
)
from smoe.utils.string_operation import str2bool

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--split_file_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering")
    parser.add_argument('--select_file_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/7B-8Expert-Select-MLP")
    parser.add_argument('--save_path', type=str, default="/home/data/models/llama-moe-transformers/7B/")
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of experts')
    parser.add_argument('--convert_type', type=str, default="LlamaMoEForCausalLM", choices=("LlamaMoEModel", "LlamaMoEForCausalLM", "LlamaMoEForSequenceClassification"))
    parser.add_argument('--use_default_gate', type=str, default="False")

    args = parser.parse_args()
    args.use_default_gate = str2bool(args.use_default_gate)
    print(args, "\n")

    if args.convert_type == "LlamaMoEModel":
        convert_llama_model_neuron_index(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            use_default_gate=args.use_default_gate
        )
    elif args.convert_type == "LlamaMoEForCausalLM":
        convert_llama_model_for_causal_lm_neuron_index(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            use_default_gate=args.use_default_gate
        )
    elif args.convert_type == "LlamaMoEForSequenceClassification":
        convert_llama_model_for_sequence_classification_neuron_index(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            use_default_gate=args.use_default_gate
        )
    else:
        raise ValueError

    # load test
    # print("Loading converted LLaMA-MoE file for test...")
    # if args.convert_type == "LlamaMoEModel":
    #     model_llama_moe = LlamaMoEModel.from_pretrained(args.save_path)
    # elif args.convert_type == "LlamaMoEForCausalLM":
    #     model_llama_moe = LlamaMoEForCausalLM.from_pretrained(args.save_path)
    # elif args.convert_type == "LlamaMoEForSequenceClassification":
    #     model_llama_moe = LlamaMoEForSequenceClassification.from_pretrained(args.save_path)
    # else:
    #     raise ValueError
    print("Done.")
