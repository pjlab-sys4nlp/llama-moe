from typing import Union


def normalize_value(value: Union[int, float], mean, std):
    if value is not None:
        if std != 0:
            return (value - mean) / std
        else:
            return value
    else:
        return None


def denormalize_value(normalized_value: Union[int, float], mean, std):
    if normalized_value is not None:
        if std != 0:
            return (normalized_value * std) + mean
        else:
            return normalized_value
    else:
        return None
