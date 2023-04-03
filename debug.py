import os, torch
# from ops.comm import setup_DDP
# from models import NeuralRecon, MnasMulti
from models.resnet import resnet18 
from torch.nn.parallel import DistributedDataParallel
# from config import cfg, update_config, args


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
    torch.distributed.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device


if __name__ == '__main__':
    rank, local_rank, world_size, device = setup_DDP(verbose=True)
    device = torch.device("cuda:{}".format(local_rank))

    if False:
        args_config = args()
        update_config(cfg, args_config)
        model = NeuralRecon(cfg)
        print(f"Build Model: rank{cfg.LOCAL_RANK}, {local_rank}, distributed {cfg.DISTRIBUTED}")
    model = resnet18(in_channels=1)
    # model = MnasMulti()
    
    model = DistributedDataParallel(
        model.to(device), 
        device_ids= [local_rank], #[cfg.LOCAL_RANK], 
        output_device=local_rank)
    print(f"Build Model: {local_rank}, ddp Completed")