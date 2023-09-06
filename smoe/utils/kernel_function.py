import torch


def pass_kernel_function(tensor, criterion):
    if criterion == "plain":
        return tensor
    elif criterion == "l1_norm":
        return torch.abs(tensor)
    elif criterion == "l2_norm":
        return tensor * tensor
    else:
        raise NotImplementedError
