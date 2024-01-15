import torch


def move_tensors_to_device(input, device):
    if isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                input[key] = value.to(device)
        return input

    elif isinstance(input, list):
        for i in range(len(input)):
            if isinstance(input[i], torch.Tensor):
                input[i] = input[i].to(device)
        return input

    elif isinstance(input, torch.Tensor):
        return input.to(device)

    else:
        raise TypeError(input)


def tensor2numbers(input):
    if isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                input[key] = value.tolist()
        return input

    elif isinstance(input, list):
        for i in range(len(input)):
            if isinstance(input[i], torch.Tensor):
                input[i] = input[i].tolist()
        return input

    elif isinstance(input, torch.Tensor):
        return input.tolist()

    else:
        raise TypeError(input)


def turn_last_true_mask_to_false(mask, true_mask_cnt=None):
    """Turn the last true value to false for each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    if true_mask_cnt is None:
        true_mask_cnt = torch.sum(mask, dim=1).unsqueeze(1)
    turn_position_indices = mask.cumsum(dim=1) == true_mask_cnt
    converted_mask = mask.clone()
    converted_mask[turn_position_indices] = False
    return converted_mask


def turn_first_true_mask_to_false(mask):
    """Turn the first true value to false for each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    turn_position_indices = mask.cumsum(dim=1) == 1
    converted_mask = mask.clone()
    converted_mask[turn_position_indices] = False
    return converted_mask


def last_true_position(mask):
    """Return the index of the last true value in each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    true_mask_cnt = torch.sum(mask, dim=1).unsqueeze(1)
    last_true_mask = (mask.cumsum(dim=1) == true_mask_cnt) & mask
    last_true_position = last_true_mask.nonzero()[:, 1].unsqueeze(1)
    return last_true_position


def pass_kernel_function(tensor, criterion):
    if criterion == "plain":
        return tensor
    elif criterion == "l1_norm":
        return torch.abs(tensor)
    elif criterion == "l2_norm":
        return tensor * tensor
    else:
        raise NotImplementedError
