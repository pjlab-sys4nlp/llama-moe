from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
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


def identity_collator(examples):  # 不对数据进行处理
    return examples


def tensor_cat_collator(examples):  # 拼接tensor
    return torch.cat(examples, dim=0)


class tensor_cat_padding_collater:  # 拼接tensor，并padding到最大长度
    def __init__(self, padding_id):
        self.padding_id = padding_id

    def __call__(self, examples):
        return rnn_utils.pad_sequence(
            examples, batch_first=True, padding_value=self.padding_id
        )


def tensor_list_cat_collator(examples):  # 拼接list中对应位置的tensor，返回list
    return [
        torch.cat([tensor[i] for tensor in examples], dim=0)
        for i in range(len(examples[0]))
    ]


class tensor_list_cat_padding_collater:  # 拼接list中对应位置的tensor，并padding到最大长度，返回list
    def __init__(self, padding_id):
        self.padding_id = padding_id

    def __call__(self, examples):
        return [
            torch.cat(
                [
                    rnn_utils.pad_sequence(
                        tensor[i], batch_first=True, padding_value=self.padding_id
                    )
                    for tensor in examples
                ],
                dim=0,
            )
            for i in range(len(examples[0]))
        ]
