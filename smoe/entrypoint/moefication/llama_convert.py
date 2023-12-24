import argparse
import os

from smoe.utils.moefication.convert_llama_moe import (
    convert_llama_model,
    convert_llama_model_for_causal_lm,
    convert_llama_model_for_sequence_classification,
)
from smoe.utils.operations.operation_string import str2bool

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--split_file_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering")
    parser.add_argument('--select_file_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/7B-8Expert-Select-MLP")
    parser.add_argument('--save_path', type=str, default="/home/data/models/llama-moe-transformers/7B/")
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of selected experts')

    parser.add_argument('--use_random_gate', type=str, default="False")
    parser.add_argument('--gate_type', type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument('--use_softmax', type=str, default='True')
    parser.add_argument('--multiply_gate_scores', type=str, default='True')

    parser.add_argument('--score_scale_factor', type=float, default=1.0, help='scale factor for experts in all layers')
    parser.add_argument('--score_scale_factor_file_path', type=str, default=None, help='file storing the layer-wise scale factors, this will override the argument "score_scale_factor"')

    parser.add_argument('--convert_type', type=str, default="LlamaMoEForCausalLM", choices=("LlamaMoEModel", "LlamaMoEForCausalLM", "LlamaMoEForSequenceClassification"))

    args = parser.parse_args()
    args.use_softmax = str2bool(args.use_softmax)
    args.multiply_gate_scores = str2bool(args.multiply_gate_scores)
    args.use_random_gate = str2bool(args.use_random_gate)
    print(args, "\n")

    if args.score_scale_factor_file_path is not None and args.score_scale_factor_file_path != "":
        with open(os.path.join(args.score_scale_factor_file_path, "score_scale_factors.txt"), "r") as file:
            layer_wise_score_scale_factor_str = file.readlines()[0]
            layer_wise_score_scale_factor = eval(layer_wise_score_scale_factor_str)
            args.score_scale_factor = layer_wise_score_scale_factor

    if args.convert_type == "LlamaMoEModel":
        convert_llama_model(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
            use_random_gate=args.use_random_gate,
            gate_type=args.gate_type,
            use_softmax=args.use_softmax,
            multiply_gate_scores=args.multiply_gate_scores,
        )
    elif args.convert_type == "LlamaMoEForCausalLM":
        convert_llama_model_for_causal_lm(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
            use_random_gate=args.use_random_gate,
            gate_type=args.gate_type,
            use_softmax=args.use_softmax,
            multiply_gate_scores=args.multiply_gate_scores,
        )
    elif args.convert_type == "LlamaMoEForSequenceClassification":
        convert_llama_model_for_sequence_classification(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
            use_random_gate=args.use_random_gate,
            gate_type=args.gate_type,
            use_softmax=args.use_softmax,
            multiply_gate_scores=args.multiply_gate_scores,
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
