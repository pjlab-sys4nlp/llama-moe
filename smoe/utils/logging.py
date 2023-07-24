import logging
import sys

import datasets
import transformers
from transformers import TrainingArguments

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
transformers.tokenization_utils.logging.set_verbosity_warning()


def set_logging(should_log, log_level):
    if should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)


def get_logger(name, log_level=None):
    logger = logging.getLogger(name)
    if log_level:
        logger.setLevel(log_level)
    return logger


def get_logger_from_training_args(name: str, training_args: TrainingArguments):
    should_log = training_args.should_log
    log_level = training_args.get_process_log_level()
    set_logging(should_log, log_level)
    logger = get_logger(name, log_level=log_level)
    return logger
