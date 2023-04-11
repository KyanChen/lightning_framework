#!/bin/bash
source ~/.bashrc
conda activate torch1mmcv2  # torch1mmcv1 torch1mmcv2 torch2mmcv1 torch2mmcv2

# pip install anypackage

cd /mnt/search01/usr/chenkeyan/codes/lightning_framework
TORCH_DISTRIBUTED_DEBUG=DETAIL python tools/train.py
# TORCH_DISTRIBUTED_DEBUG=DETAIL
#python train.py
#python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train.py
#python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_pipe.py
# juicesync src dst
# juicefs rmr your_dir