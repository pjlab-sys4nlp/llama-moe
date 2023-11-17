import os
import socket

import torch
import torch.distributed as dist
import torch.nn as nn

# from accelerate import Accelerator


def test_connection():
    string = f"{socket.gethostname()} - MASTER_ADDR: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} - WORLD_SIZE: {os.environ['WORLD_SIZE']} - RANK: {os.environ['RANK']}"
    print(string)
    dist.init_process_group("nccl")
    # ac = Accelerator()
    m = nn.Linear(5, 10)
    m = nn.parallel.DistributedDataParallel(m, device_ids=[dist.get_rank()])
    # m = ac.prepare_model(m)
    x = torch.randn(3, 5, device=m.device)
    y = m(x)
    # dist.all_reduce(y, op=dist.ReduceOp.SUM)
    assert y.shape == (3, 10)
    # print(f"Done - local: {ac.local_process_index} - rank: {ac.process_index} - world: {ac.num_processes}")
    print(
        f"Done - {socket.gethostname()} - local: {os.environ['LOCAL_RANK']} - rank: {dist.get_rank()} - world: {dist.get_world_size()}"
    )


if __name__ == "__main__":
    test_connection()
