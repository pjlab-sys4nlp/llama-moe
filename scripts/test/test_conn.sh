#!/usr/bin/bash

#SBATCH --job-name=test_conn
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=3
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

export OMP_NUM_THREADS=4

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}

srun torchrun \
    --nnodes 3 \
    --nproc_per_node 8 \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29520 \
    tests/entrypoint/test_conn.py
