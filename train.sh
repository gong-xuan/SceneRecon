DEVICES='6,7'

# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port 29555 train.py
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=2 --master_port 29555 train.py

CUDA_VISIBLE_DEVICES=7 python train.py --pretrain
# NCLL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 train.py --pretrain
# PL_TORCH_DISTRIBUTED_BACKEND=gloo 
# NCLL_P2P_DISABLE=1 \
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 debug.py


# CUDA_VISIBLE_DEVICES=1 python train.py

#Run test
# NCLL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 test.py
# CUDA_VISIBLE_DEVICES=1 python test.py --eps 50 --epe 51
# python eval.py --n_proc 2 --gpu 0,1,2,3,4

python test.py --eps 50 --epe 51 --gpu 1