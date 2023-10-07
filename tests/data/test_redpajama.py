import time
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

from smoe.data.redpajama import load_streaming_datasets
from smoe.utils.io import dump_jsonlines, load_jsonlines


def test_load_streaming_datasets():
    output_dir = Path("/mnt/petrelfs/zhutong/smoe/resources/data_test_with_task_type")
    output_dir.mkdir(parents=True, exist_ok=True)
    # dataset_dir = Path("resources/data_test")
    dataset_dir = Path("resources/data_test_with_task_type")

    # # update new dataset with task type
    # for subtask_dir in dataset_dir.glob("*"):
    #     task_type = subtask_dir.stem
    #     subtask_out_dir = output_dir.joinpath(task_type)
    #     subtask_out_dir.mkdir(parents=True, exist_ok=True)
    #     for file in subtask_dir.glob("*.jsonl"):
    #         data = load_jsonlines(file)
    #         for ins in data:
    #             ins["src"] = task_type
    #         dump_jsonlines(data, subtask_out_dir.joinpath(file.name))

    dataset = load_streaming_datasets(
        str(dataset_dir),
        prob_map={"en_arxiv": 0.5, "en_book": 0.2, "en_c4": 0.3},
        block_size=2048,
    )
    num_ds = 0
    num_src = defaultdict(lambda: 0)

    start = time.time()
    for ds in iter(dataset):
        num_ds += 1
        # print(num_ds, ds["src"])
        # num_src[ds["src"]] += 1
    time_span = time.time() - start
    print(num_ds)
    print(dict(num_src))
    print(f"Time (ins/s): {num_ds / time_span:.2f}" "")

    """
    block_size: -1
    {'en_arxiv': 400, 'en_c4': 214}
    Time (ins/s): 64.05

    block_size: 2048
    Time (ins/s): 59.94
    """


if __name__ == "__main__":
    test_load_streaming_datasets()
