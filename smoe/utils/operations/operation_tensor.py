import torch

def move_tensors_to_device(input_dict, device):
    """
    Move all tensors in the input dictionary to the specified device.

    Args:
    - input_dict (dict): Dictionary containing tensors.
    - device (str or torch.device): Target device, e.g., 'cpu' or 'cuda:0'.

    Returns:
    - dict: New dictionary with all tensors moved to the target device.
    """
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.to(device)
        else:
            input_dict[key] = value

    return input_dict


def turn_last_true_mask_to_false(mask, true_mask_cnt=None):
    """Turn the last true value to false for each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    if true_mask_cnt is None:
        true_mask_cnt = torch.sum(mask, dim=1).unsqueeze(1)
    turn_position_indices = (mask.cumsum(dim=1) == true_mask_cnt)
    converted_mask = mask.clone()
    converted_mask[turn_position_indices] = False
    return converted_mask


def turn_first_true_mask_to_false(mask):
    """Turn the first true value to false for each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    turn_position_indices = (mask.cumsum(dim=1) == 1)
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
