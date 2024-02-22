import logging
import os
import socket
import sys
from pathlib import Path

import datasets
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from smoe.callbacks.save_model import SchedulerStateCallback
from smoe.callbacks.tensorboard import EnhancedTensorboardCallback
from smoe.data.collate_fn import fault_tolerance_data_collator
from smoe.data.dynamic_selection import (
    AVERAGE_SLIMPAJAMA_DATA_PORTION,
    LLAMA_DATA_PORTION,
    SHEAREDLLAMA_DATA_PORTION,
)
from smoe.data.streaming import CachedJsonlDataset, SubDirWeightedPackedJsonlDataset
from smoe.metrics.preprocess import logits_argmax
from smoe.models.configuration_llama_moe import LlamaMoEConfig
from smoe.models.modeling_llama_moe_hf import LlamaMoEForCausalLM
from smoe.modules.flash_attn import replace_xformers
from smoe.trainer.llama_lr_scheduling import LlamaLrSchedulingTrainer
from smoe.utils.config import (
    DataArguments,
    EnhancedTrainingArguments,
    ModelArguments,
    parse_args,
)
from smoe.utils.param import get_trainable_parameters

MODEL_MAP = {
    "llama": LlamaForCausalLM,
    "llama_moe": LlamaMoEForCausalLM,
}

CONFIG_MAPPING.update(
    {
        "llama": LlamaConfig,
        "llama_moe": LlamaMoEConfig,
    }
)

logger = logging.getLogger(__name__)


# @wechat_sender(msg_prefix="CPT Training")
def main():
    model_args, data_args, training_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    hostname = socket.gethostname()
    logger.warning(
        f"Global rank: {training_args.process_index}, "
        f"Host: {hostname}, IP: {socket.gethostbyname(hostname)}, "
        f"Process local rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"fp16 training: {training_args.fp16}, "
        f"bf16 training: {training_args.bf16}"
    )
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args.to_json_string()}")

    if training_args.debug_mode:
        from smoe.utils.debugging import remote_breakpoint

        remote_breakpoint()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is"
                " not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid"
                " this behavior, change the `--output_dir` or add"
                " `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    logger.info(f"Seed set to: {training_args.seed}")
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "gate_type": model_args.gate_type,
        "calculator_type": model_args.calculator_type,
        "num_selects": model_args.num_selects,
        "gate_network": model_args.gate_network_type,
        "score_scale_factor": model_args.moe_calculator_score_scale_factor,
        "gate_balance_loss_weight": model_args.gate_balance_loss_weight,
    }
    if model_args.config_name == "llama_moe" or model_args.model_type == "llama_moe":
        ConfigClass = LlamaMoEConfig
    else:
        ConfigClass = AutoConfig

    if model_args.config_name:
        config = ConfigClass.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ConfigClass.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if training_args.gradient_checkpointing:
        config.use_cache = False

    # zhutong: this is for debug usage only
    if training_args.debug_mode:
        config.num_hidden_layers = 2

    if model_args.model_type == "mixtral" or model_args.model_name_or_path == "mixtral":
        config.num_experts_per_tok = model_args.num_selects
        config.output_router_logits = True

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "legacy": True if model_args.use_legacy_tokenizer else False,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported"
            " by this script.You can do it from another script, save it, and load it"
            " from here, using --tokenizer_name."
        )

    # Preprocessing the datasets.
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than"
                " the default `block_size` value of 1024. If you would like to use a"
                " longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the"
                f" maximum length for the model({tokenizer.model_max_length}). Using"
                f" block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    prob_map = LLAMA_DATA_PORTION
    if data_args.prob_map == "uniform":
        prob_map = AVERAGE_SLIMPAJAMA_DATA_PORTION
    elif data_args.prob_map == "sheared_llama":
        prob_map = SHEAREDLLAMA_DATA_PORTION

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = SubDirWeightedPackedJsonlDataset(
            data_args.dataset_dir,
            prob_map=prob_map,
            seed=training_args.seed,
            block_size=data_args.block_size,
        )
        # lm_datasets = load_streaming_datasets(
        #     data_args.dataset_dir,
        #     prob_map=data_args.prob_map,
        #     num_proc=data_args.preprocessing_num_workers,
        #     debug_mode=training_args.debug_mode,
        #     block_size=data_args.block_size,
        # )

    if training_args.do_train:
        train_dataset = lm_datasets
        if data_args.max_train_samples is None:
            raise ValueError("max_train_samples cannot be None")
        logger.info("training example:")
        res = None
        if hasattr(train_dataset, "take"):
            res = tokenizer.decode([x["input_ids"] for x in train_dataset.take(1)][0])
        else:
            for x in train_dataset:
                input_ids = x["input_ids"]
                break
            res = tokenizer.decode(input_ids)
        logger.info(res)

    eval_dataset = None
    if training_args.do_eval:
        paths = Path(data_args.validation_dir).glob("*.jsonl")
        eval_dataset = {
            path.stem: CachedJsonlDataset(
                str(path), training_args.seed, block_size=data_args.block_size
            )
            for path in paths
        }
        logger.info(f"eval types: {list(eval_dataset.keys())}")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        ModelClass = MODEL_MAP[model_args.model_type]

        # model = LlamaForCausalLM(config)
        # model.half()
        # model.to(torch_dtype)

        model: LlamaForCausalLM | LlamaMoEForCausalLM = ModelClass.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # train an MoE model from scratch 👇
        # config.num_hidden_layers = 20
        # model: LlamaMoEForCausalLM = LlamaMoEForCausalLM(config)
        # if isinstance(model, LlamaMoEForCausalLM):
        #     for name, param in model.named_parameters():
        #         if "weight_noise.weight" in name:
        #             nn.init.zeros_(param)
        #     model.change_moe_gate_add_noise(True)
        #     model.change_moe_gate_use_balance(True)
        # model.reset_gate_network()
        replace_xformers(model)
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params"
        )

    # for name, param in model.named_parameters():
    #     # if ".mlp_norm." not in name:
    #     if ".gate." not in name and ".mlp_norm." not in name:
    #         param.requires_grad = False
    #     logger.info(f"{name} ({param.numel()}) - Grad: {param.requires_grad}")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # if hasattr(model, "set_moe_calculator_score_scale_factor"):
    #     # update config for checkpoint retrival
    #     # model.set_moe_gate_balance_loss_weight(0.1)
    #     # model.set_moe_calculator_score_scale_factor(4.0)
    #     model.set_moe_calculator_score_scale_factor(
    #         model_args.moe_calculator_score_scale_factor
    #     )
    #     # model.set_moe_calculator_score_scale_factor(1.0)
    #     model.update_config()

    model_vocab_size = model.get_output_embeddings().weight.size(0)
    if model_vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        raise ValueError(
            f"The model's vocab size ({model_vocab_size}) does not match with the"
            f" tokenizer ({len(tokenizer)})"
        )

    trainable_params, _ = get_trainable_parameters(model, verbose=True)
    training_args.num_training_params = trainable_params

    # Initialize our Trainer
    trainer = LlamaLrSchedulingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=(
            logits_argmax
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
    )
    trainer.add_callback(EnhancedTensorboardCallback)
    trainer.add_callback(SchedulerStateCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = data_args.max_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        if isinstance(trainer.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
                dataset_metrics = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=None,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(ignore_keys=None)
        logger.info(f"{metrics}")


if __name__ == "__main__":
    main()
