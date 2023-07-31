import torch.nn as nn

from smoe.utils.logging import get_logger

logger = get_logger(__name__)


def get_trainable_parameters(model: nn.Module, verbose: bool = True):
    """
    Prints the number of trainable parameters in the model.

    Credit to https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if verbose:
        logger.info(
            f"trainable params: {trainable_params:,d}"
            f" || all params: {all_param:,d}"
            f" || trainable%: {100 * trainable_params / all_param}"
        )

    return trainable_params, all_param
