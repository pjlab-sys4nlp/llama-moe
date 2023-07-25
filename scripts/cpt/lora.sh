#!/usr/bin/bash

#SBATCH --job-name=cpt-lora-speed
#SBATCH --partition=MoE
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.log

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --nodes=1
#SBATCH --gres=gpu:1

source ~/anaconda3/bin/activate torch

num_nodes=1         # should match with --nodes
num_gpu_per_node=1  # should match with --gres

# #cpu/#num_gpu_per_node
export OMP_NUM_THREADS=1

lr=2e-4
lora_rank=8
lora_alpha=32
lora_dropout=0.05
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"

pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B/
tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B/
dataset_dir=/mnt/petrelfs/share_data/quxiaoye/pretrain_LLAMA_all_data_processed
per_device_train_batch_size=24
per_device_eval_batch_size=1
gradient_accumulation_steps=1
block_size=2048
max_steps=$(echo "10^11 / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
max_train_samples=$(echo "10^11 / $block_size" | bc)
echo "max_steps: $max_steps"
echo "max_train_samples: $max_train_samples"
global_bs=$(echo "$per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node" | bc)
echo "global batch size: $global_bs"
tokens_per_batch=$(echo "$global_bs * $block_size" | bc)
echo "#tokens/batch: $tokens_per_batch"

data_cache=resources/cache
output_dir=outputs/cpt-lora-bf16-4nodes
deepspeed_config_file=conf/deepspeed/bf16.json

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIS ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"
export LOGLEVEL=INFO

srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29518 \
    smoe/entrypoint/cpt_lora.py \
        --deepspeed ${deepspeed_config_file} \
        --model_name_or_path ${pretrained_model} \
        --tokenizer_name_or_path ${tokenizer_path} \
        --dataset_dir ${dataset_dir} \
        --data_cache_dir ${data_cache} \
        --validation_split_percentage 0.001 \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --do_train \
        --seed $RANDOM \
        --bf16 \
        --num_train_epochs 1 \
        --final_lr_portion 0.1 \
        --optim adamw_torch \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate ${lr} \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --warmup_steps 2000 \
        --max_steps 48828125 \
        --max_train_samples 48828125 \
        --logging_strategy steps \
        --logging_steps 10 \
        --save_strategy steps \
        --save_total_limit 3 \
        --save_steps 1000 \
        --dataloader_num_workers 1 \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --block_size ${block_size} \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --lora_rank ${lora_rank} \
        --lora_alpha ${lora_alpha} \
        --trainable ${lora_trainable} \
        --lora_dropout ${lora_dropout} \
        --torch_dtype auto \
        --ddp_find_unused_parameters False \
        --report_to tensorboard

        # --gradient_checkpointing \
        # --modules_to_save ${modules_to_save} \
