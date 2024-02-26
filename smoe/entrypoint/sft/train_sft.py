import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import transformers
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, Trainer
from transformers.trainer_pt_utils import LabelSmoother

from smoe.utils.conversation import Conversation
from smoe.utils.io import load_json, load_jsonlines

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_type: str = field(
        default="auto", metadata={"help": "Model type: `moe` or `mixtral` or `auto`"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype: `float32` or `bfloat16`"},
    )
    additional_config: str = field(
        default=None,
        metadata={"help": "Additional config file (in json) to load"},
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": "attention implementation, choice from [eager, flash_attention_2, sdpa] (default: `flash_attention_2`)"
        },
    )

    def __post_init__(self):
        if hasattr(torch, self.torch_dtype):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        if self.additional_config is not None:
            if not pathlib.Path(self.additional_config).exists():
                raise ValueError(
                    f"Additional config file {self.additional_config} not found"
                )
            self.additional_config = load_json(self.additional_config)


@dataclass
class DataArguments:
    eval_data_dir: str = field(
        default=None, metadata={"help": "Path to the evaluation data folder."}
    )
    dataset_dir_or_path: str = field(
        default="data/merged",
        metadata={"help": "Path to dataset directory or a single jsonl file"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the gate during training."},
    )
    save_final_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save final checkpoint."},
    )


def trainer_save_model_safe(trainer):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    instances,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    tokenizer_legacy = getattr(tokenizer, "legacy", True)
    conv = Conversation()
    conv.sep2 = tokenizer.eos_token
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, ins in enumerate(instances):
        if roles[ins["conversations"][0]["from"]] != roles["human"]:
            # Skip the first one if it is not from human
            ins["conversations"] = ins["conversations"][1:]

        conv.clear_msg()
        sys_msg = ins.get("system_prompt")
        if sys_msg is not None:
            conv.set_system_message(sys_msg)
        else:
            conv.set_system_message("")
        for j, turn in enumerate(ins["conversations"]):
            role = roles[turn["from"]]
            assert role == conv.roles[j % 2], f"{i}/{j}"
            conv.append_message(role, turn["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    res = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = res["input_ids"]
    attention_masks = res["attention_mask"]
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    # attention_masks = torch.ones_like(input_ids)
    for conversation, target, attention_mask in zip(
        conversations, targets, attention_masks
    ):
        turns = conversation.split(conv.sep2)
        # the eos token is included in `total_len`, llama2 will add bos token
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + len(turns) * int(tokenizer.pad_token == tokenizer.eos_token)
        # attention_mask[total_len:] = 0
        total_len = attention_mask.sum()

        cur_len = 0
        has_bos = False
        if target[0] == tokenizer.bos_token_id:
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID  # bos token
            has_bos = True
        for i, turn in enumerate(turns):
            if turn == "":
                break
            # +1: add sep2 token
            turn_len = len(tokenizer(turn).input_ids) - int(has_bos) + 1

            # sep: " ASSISTANT: "
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct: bos and the last space token
            # -1 means remove extra suffix space in sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - int(has_bos) - 1

            if i != 0 and not tokenizer_legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
            # if i < len(turns) - 1:
            #     # plus one for sep2 token (eos)
            #     cur_len += 1

            if i != 0 and not tokenizer_legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                logger.info(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks,
    )


def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        try:
            features = [vars(f) for f in features]
        except TypeError:
            print(len(features), type(features[0]), features[0])
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


class CachedJsonlDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        tokenizer: PreTrainedTokenizer,
        seed: int = 1227,
    ) -> None:
        super().__init__()
        self.datapath = datapath
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.data = load_jsonlines(datapath)
        self.rng.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        ins = self.data[index]
        processed = preprocess([ins], self.tokenizer)
        ins = {}
        for key in processed:
            ins[key] = processed[key][0]
        return ins

    def state_dict(self):
        return {
            "datapath": self.datapath,
            "seed": self.seed,
            "rng": self.rng.getstate(),
        }


def get_tokenizer(
    model_name_or_path,
    cache_dir: str = None,
    model_max_length: int = 2048,
    padding_side: str = "right",
    use_fast: bool = False,
    trust_remote_code: bool = False,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer ready, pad_token: {tokenizer.pad_token}")
    return tokenizer


def get_model(
    model_type: str,
    model_name_or_path: str,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    additional_config: dict = None,
):
    logger.info(f"Model type: {model_type}")
    if model_type == "auto":
        ConfigClass = transformers.AutoConfig
        ModelClass = transformers.AutoModelForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set RoPE scaling factor
    config = ConfigClass.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    if additional_config is not None:
        config.update(additional_config)
    logger.info("Config ready")

    # Load model and tokenizer
    model = ModelClass.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )
    logger.info("model ready")

    return model


def get_model_and_tokenizer(
    model_type: str,
    model_name_or_path: str,
    tokenizer_path: str = None,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    padding_side: str = "right",
    additional_config: dict = None,
    use_fast: bool = False,
) -> tuple:
    if tokenizer_path is None:
        tokenizer_path = model_name_or_path
    tokenizer = get_tokenizer(
        tokenizer_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    model = get_model(
        model_type,
        model_name_or_path,
        torch_dtype=torch_dtype,
        model_max_length=model_max_length,
        attn_impl=attn_impl,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        additional_config=additional_config,
    )

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    model, tokenizer = get_model_and_tokenizer(
        model_args.model_type,
        model_args.model_name_or_path,
        tokenizer_path=model_args.tokenizer_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        torch_dtype=model_args.torch_dtype,
        additional_config=model_args.additional_config,
        attn_impl=model_args.attn_impl,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.freeze_gate:
        for name, param in model.named_parameters():
            if "gate" in name:
                param.requires_grad = False

    train_dataset = None
    datapath = pathlib.Path(data_args.dataset_dir_or_path)
    if not datapath.exists():
        raise ValueError(f"Dataset path {datapath} not found")
    elif datapath.is_file():
        logger.info(f"CachedJsonlDataset: {datapath}")
        train_dataset = CachedJsonlDataset(
            data_args.dataset_dir_or_path,
            tokenizer,
            seed=training_args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset path type: {datapath}")
    logger.info("train dataset ready")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=fault_tolerance_data_collator,
    )
    logger.info("trainer ready")

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logger.info("resume training from ckpt")
            trainer.train(resume_from_checkpoint=True)
        else:
            logger.info("start training")
            trainer.train()

    # Save model
    if training_args.save_final_ckpt:
        logger.info("training finished, dumping model")
        model.config.use_cache = True
        trainer.save_state()
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)

    logger.info("🎉 All done~")


if __name__ == "__main__":
    train()
