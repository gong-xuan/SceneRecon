"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import torch.distributed as dist
import os
import torch
import datetime

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    print(world_size)
    if world_size == 1:
        return
    dist.barrier()


def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend,
                            # init_method='env://',
                            # timeout=datetime.timedelta(seconds=5400)
                            )
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device