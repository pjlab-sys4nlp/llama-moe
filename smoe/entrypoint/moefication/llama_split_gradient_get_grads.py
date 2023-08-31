import os.path
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, set_seed

from smoe.data.collate_fn import fault_tolerance_data_collator
from smoe.data.single_file import load_cached_dataset
from smoe.trainer.moefication.expert_split_gradient import ExpertSplitGradientTrainer
from smoe.utils.change_llama_forward import forward_llama_mlp_with_backward_hook_bug_fix
from smoe.utils.config import (
    DataArguments,
    EnhancedTrainingArguments,
    ModelArguments,
    parse_args,
)
from smoe.utils.moefication.expert_split import GradientSplitGetGrads
from smoe.utils.param import get_trainable_parameters


@dataclass
class SplitArguments:
    save_path: str = field(default=None)
    expert_size: int = field(default=None)
    kernel: Optional[str] = field(default="l1_norm")  # plain l1_norm l2_norm
    accumulate_level: Optional[str] = field(default="sample")  # sample total
    data_use_range_begin: Optional[float] = field(default=0.0)
    data_use_range_end: Optional[float] = field(default=1.0)
    importance_type: Optional[str] = field(
        default="feature_grad"
    )  # feature_grad feature_change


@record
def main():
    model_args, data_args, training_args, split_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments, SplitArguments
    )
    model_name = os.path.split(model_args.model_name_or_path)[1]
    dataset_name = os.path.split(data_args.dataset_dir)[1].split(".")[0]
    split_args.save_path = os.path.join(
        split_args.save_path,
        "Gradients",
        f"{model_name}-Gradients-{split_args.kernel}-{split_args.accumulate_level}-{split_args.importance_type}",
        dataset_name,
    )
    print(split_args, "\n")

    """Set seed before initializing model."""
    set_seed(training_args.seed)

    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    if training_args.gradient_checkpointing:
        config.use_cache = False

    tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    """Preprocessing the datasets."""
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        block_size = 2048 if block_size > 2048 else block_size
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = load_cached_dataset(
            data_args.dataset_dir,
            block_size=block_size,
        )

    """Initialize model"""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # locate block by the name template
    for layer_index, layer in enumerate(model.model.layers):
        layer.mlp.forward = types.MethodType(
            forward_llama_mlp_with_backward_hook_bug_fix, layer.mlp
        )

    model_vocab_size = model.get_output_embeddings().weight.size(0)
    if model_vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        raise ValueError(
            f"The model's vocab size ({model_vocab_size}) does not match with the"
            f" tokenizer ({len(tokenizer)})"
        )

    get_trainable_parameters(model, verbose=True)

    """Initialize our Trainer"""
    trainer = ExpertSplitGradientTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
    )

    """Start splitting"""
    split = GradientSplitGetGrads(
        split_args,
        trainer,
        accumulate_level=split_args.accumulate_level,
        kernel=split_args.kernel,
        importance_type=split_args.importance_type,
        device=f"cuda:{training_args.local_rank}",
    )
    split.get_score()
    print(f"Device {training_args.local_rank} Done.")


if __name__ == "__main__":
    main()
