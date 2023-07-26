import argparse

from smoe.models.llama_moefication.modeling_llama_moe import (
    LlamaMoEForCausalLM,
    LlamaMoEForSequenceClassification,
    LlamaMoEModel,
)
from smoe.utils.moefication.convert_llama_moe import (
    convert_llama_model,
    convert_llama_model_for_causal_lm,
    convert_llama_model_for_sequence_classificaiton,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/home/data/models/llama-transformers/7B"
    )
    parser.add_argument(
        "--split_file_path",
        type=str,
        default="/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering",
    )
    parser.add_argument(
        "--select_file_path",
        type=str,
        default="/home/dongdz/workspace/moefication/llama_moe_temp_files/7B-8Expert-Select-MLP",
    )
    parser.add_argument(
        "--save_path", type=str, default="/home/data/models/llama-moe-transformers/7B/"
    )
    parser.add_argument(
        "--templates", type=str, default="layers.{}.mlp.gate_proj.weight"
    )
    parser.add_argument("--num_experts", type=int, default=8, help="number of experts")
    parser.add_argument("--num_selects", type=int, default=2, help="number of experts")
    parser.add_argument(
        "--convert_type",
        type=str,
        default="LlamaMoEForCausalLM",
        choices=(
            "LlamaMoEModel",
            "LlamaMoEForCausalLM",
            "LlamaMoEForSequenceClassification",
        ),
    )

    args = parser.parse_args()

    if args.convert_type == "LlamaMoEModel":
        convert_llama_model(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.templates,
            args.num_experts,
            args.num_selects,
        )
    elif args.convert_type == "LlamaMoEForCausalLM":
        convert_llama_model_for_causal_lm(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.templates,
            args.num_experts,
            args.num_selects,
        )
    elif args.convert_type == "LlamaMoEForSequenceClassification":
        convert_llama_model_for_sequence_classificaiton(
            args.model_path,
            args.split_file_path,
            args.select_file_path,
            args.save_path,
            args.templates,
            args.num_experts,
            args.num_selects,
        )
    # load test
    print("Loading converted LLaMA-MoE file for test...")
    if args.convert_type == "LlamaMoEModel":
        model_llama_moe = LlamaMoEModel.from_pretrained(args.save_path)
    elif args.convert_type == "LlamaMoEForCausalLM":
        model_llama_moe = LlamaMoEForCausalLM.from_pretrained(args.save_path)
    elif args.convert_type == "LlamaMoEForSequenceClassification":
        model_llama_moe = LlamaMoEForSequenceClassification.from_pretrained(
            args.save_path
        )
    print("Done.")
