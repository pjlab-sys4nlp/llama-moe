import argparse

from smoe.utils.moefication.convert_llama_moe_neuron_index_residual import (
    convert_llama_model_for_causal_lm_neuron_index_residual,
    convert_llama_model_for_sequence_classification_neuron_index_residual,
    convert_llama_model_neuron_index_residual,
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

    parser.add_argument('--num_experts', type=int, default=14, help='number of moe experts')
    parser.add_argument('--num_experts_residual', type=int, default=2, help='number of residual experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of selected moe experts')
    parser.add_argument('--score_scale_factor', type=float, default=1.0, help='scale factor for moe experts in all layers')
    parser.add_argument('--score_scale_factor_residual', type=float, default=1.0, help='scale factor for residual experts in all layers')
    # parser.add_argument('--score_scale_factor_file_path', type=str, default=None, help='file storing the layer-wise scale factors, this will override the arguments "score_scale_factor" and "score_scale_factor_residual"')

    parser.add_argument('--convert_type', type=str, default="LlamaMoEResidualForCausalLM", choices=("LlamaMoEResidualModel", "LlamaMoEResidualForCausalLM", "LlamaMoEResidualForSequenceClassification"))
    parser.add_argument('--use_default_gate', type=str, default="False")

    args = parser.parse_args()
    args.use_default_gate = str2bool(args.use_default_gate)
    print(args, "\n")

    if args.convert_type == "LlamaMoEResidualModel":
        convert_llama_model_neuron_index_residual(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_experts_residual,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
            score_scale_factor_residual=args.score_scale_factor_residual,
            use_default_gate=args.use_default_gate
        )
    elif args.convert_type == "LlamaMoEResidualForCausalLM":
        convert_llama_model_for_causal_lm_neuron_index_residual(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_experts_residual,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
            score_scale_factor_residual=args.score_scale_factor_residual,
            use_default_gate=args.use_default_gate
        )
    elif args.convert_type == "LlamaMoEResidualForSequenceClassification":
        convert_llama_model_for_sequence_classification_neuron_index_residual(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_experts_residual,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
            score_scale_factor_residual=args.score_scale_factor_residual,
            use_default_gate=args.use_default_gate
        )
    else:
        raise ValueError

    # load test
    # print("Loading converted LLaMA-MoE file for test...")
    # if args.convert_type == "LlamaMoEModel":
    #     model_llama_moe = LlamaMoEResidualModel.from_pretrained(args.save_path)
    # elif args.convert_type == "LlamaMoEForCausalLM":
    #     model_llama_moe = LlamaMoEResidualForCausalLM.from_pretrained(args.save_path)
    # elif args.convert_type == "LlamaMoEForSequenceClassification":
    #     model_llama_moe = LlamaMoEResidualForSequenceClassification.from_pretrained(args.save_path)
    # else:
    #     raise ValueError
    print("Done.")
