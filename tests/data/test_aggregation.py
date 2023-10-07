from smoe.data.aggregation import group_instances


def test_group_instances():
    instances = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
    ]
    results = group_instances(instances, block_size=4)
    assert results == [
        {"input_ids": [1, 2, 3, 1], "labels": [4, 5, 6, 4]},
        {"input_ids": [2, 3, 1, 2], "labels": [5, 6, 4, 5]},
        {"input_ids": [3, 1, 2, 3], "labels": [6, 4, 5, 6]},
    ]


if __name__ == "__main__":
    test_group_instances()
