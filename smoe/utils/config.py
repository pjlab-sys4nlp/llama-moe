import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, TypeVar

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want"
                " to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to"
                " train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If training from scratch, pass a model type from the list: "
                + ", ".join(MODEL_TYPES)
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained"
                " from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded from"
                " huggingface.co"
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer (backed by the tokenizers"
                " library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name or"
                " commit id)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login`"
                " (necessary to use this script with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this"
                " dtype. If `auto` is passed, the dtype will be automatically derived"
                " from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    gate_type: Literal["TopKBalancedNoisyGate", "SwitchBalancedGate"] = field(
        default="TopKBalancedNoisyGate",
        metadata={
            "help": "The type of gate, should be one of `TopKBalancedNoisyGate` and `SwitchBalancedGate`"
        },
    )
    calculator_type: Literal[
        "UniversalCalculator", "SwitchDropTokenCalculator"
    ] = field(
        default="UniversalCalculator",
        metadata={
            "help": "The type of gate calculator, should be one of `UniversalCalculator` and `SwitchDropTokenCalculator`"
        },
    )
    num_selects: int = field(
        default=4, metadata={"help": "The number of experts to be selected"}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or"
                " --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use (via the datasets"
                " library)."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the perplexity on"
                " (a text file)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " evaluation examples to this value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. The training"
                " dataset will be truncated in block of this size for training. Default"
                " to the model max input length for single sentence inputs (take into"
                " account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": (
                "The percentage of the train set used as validation set in case there's"
                " no validation split"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    data_cache_dir: Optional[str] = field(
        default="./", metadata={"help": "The datasets processed stored"}
    )
    prob_map: Optional[dict[str, float]] = field(
        default=None,
        metadata={
            "help": (
                'data type to sampling probabilities. e.g. {"commoncrawl": 0.67, "c4":'
                " 0.15}"
            )
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    final_lr_portion: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Final lr = learning_rate * final_lr_portion. Default is 0.0"
        },
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to True, the number of data files will be cut to 2."},
    )
    _block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "This is an automatic assigned value from DataArguments, and should not be hijacked. In most cases, you should not use this argument manually."
        },
    )
    num_tokens_per_batch: Optional[int] = field(
        default=-1,
        metadata={
            "help": "the number of tokens per batch, should be calculated automatically after `block_size`"
        },
    )

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                "`block_size` should be an integer that is greater than 0!"
            )
        self._block_size = value
        self.num_tokens_per_batch = (
            self._block_size
            * self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )


@dataclass
class LoraTrainingArguments(EnhancedTrainingArguments):
    trainable: Optional[str] = field(default="q_proj,v_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)


Arguments = TypeVar("Arguments")


def parse_args(*args: Type[Arguments]) -> tuple[Arguments, ...]:
    """
    Parse arguments from different argument dataclasses

    Example:
        >>> model_args, data_args, train_args = parse_args(ModelArguments, DataArguments, TrainingArguments)
    """
    parser = HfArgumentParser(args)
    if len(sys.argv) == 2:
        if sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            arg_tuple = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        elif sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
            arg_tuple = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
        else:
            raise ValueError(
                "Only yaml, yml, and json config files are supported, got"
                f" {sys.argv[1]}"
            )
    else:
        arg_tuple = parser.parse_args_into_dataclasses()

    data_args = None
    training_args = None
    for arg in arg_tuple:
        if isinstance(arg, DataArguments):
            data_args = arg
        elif isinstance(arg, TrainingArguments):
            training_args = arg

    if hasattr(data_args, "block_size"):
        training_args.block_size = data_args.block_size

    return arg_tuple
