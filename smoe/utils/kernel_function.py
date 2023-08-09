import torch

available_choices = ("plain", "l1_norm", "l2_norm")


def pass_kernel_function(tensor, criterion):
    assert criterion in available_choices

    if criterion == "plain":
        return tensor
    elif criterion == "l1_norm":
        return torch.abs(tensor)
    elif criterion == "l2_norm":
        return tensor * tensor
