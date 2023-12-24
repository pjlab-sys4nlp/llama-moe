# !/usr/bin/bash

# SBATCH --job-name=test_conn
# SBATCH --output=logs/test_conn.log
# SBATCH --error=logs/test_conn.log

# SBATCH --partition=MoE_T
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=26
# SBATCH --mem=0

# SBATCH --nodes=8
# SBATCH --gres=gpu:1
# SBATCH --quotatype=reserved

# srun -p MoE_T -N8 -n8 --gres=gpu:1 -w HOST-10-140-60-[134,141,163,180-181,184] torchrun --nnodes 8 --nproc_per_node 1 tests/entrypoint/test_conn.py
# $ srun -p MoE_T -N8 -n8 --gres=gpu:1 -w HOST-10-140-60-[134,141,163,180-181,184] bash scripts/cpt/test/test_conn.sh

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"
echo "Node list: $nodes"

torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29519 \
    tests/entrypoint/test_conn.py
