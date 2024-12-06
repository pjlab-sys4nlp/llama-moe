import torch


def move_tensors_to_device(input, device):
    if input is None:
        return input

    elif isinstance(input, dict):
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


def tensors_are_same(tensor1, tensor2):
    if type(tensor1) != type(tensor2):
        return False

    elif isinstance(tensor1, dict):
        if tensor1.keys() != tensor2.keys():
            return False
        for key in tensor1:
            if not torch.equal(tensor1[key], tensor2[key]):
                return False
        return True

    elif isinstance(tensor1, list):
        if len(tensor1) != len(tensor2):
            return False
        for item1, item2 in zip(tensor1, tensor2):
            if not torch.equal(item1, item2):
                return False
        return True

    elif isinstance(tensor1, torch.Tensor):
        return torch.equal(tensor1, tensor2)

    else:
        raise TypeError("Unsupported tensor types:", type(tensor1), type(tensor2))


def tensor2numbers(input):
    if input is None:
        return input

    elif isinstance(input, dict):
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


def last_true_position(mask):
    """Return the index of the last true value in each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    true_mask_cnt = torch.sum(mask, dim=1).unsqueeze(1)
    last_true_mask = (mask.cumsum(dim=1) == true_mask_cnt) & mask
    last_true_position = last_true_mask.nonzero()[:, 1].unsqueeze(1)
    return last_true_position


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


def equalize_true_in_mask(mask, max_true_threshold):
    """
    Adjust the mask such that each row has the same number of True values,
    specified by max_true_threshold, by turning the last few True values to False if a row exceeds the threshold.
    """
    # mask: shape(batch_size, seq_len)
    retain_true_positions = mask.cumsum(dim=1) <= max_true_threshold
    mask = mask & retain_true_positions
    return mask


def pass_kernel_function(tensor, criterion, allow_nan=False):
    if criterion == "plain":
        return tensor
    elif criterion == "sqrt":
        if not allow_nan and torch.any(tensor < 0):
            raise ValueError(
                'Detected negative value in the tensor! This will cause the result to be "nan"!'
            )
        return torch.sqrt(tensor)
    elif criterion == "l1":
        return torch.abs(tensor)
    elif criterion == "l2":
        return tensor * tensor
    else:
        raise NotImplementedError
