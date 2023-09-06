#!/usr/bin/bash

#SBATCH --job-name=cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -x SH-IDCA1404-10-140-54-116

#SBATCH --nodes=8
#SBATCH --gres=gpu:8

source ~/anaconda3/bin/activate smoe

num_nodes=8         # should match with --nodes
num_gpu_per_node=8  # should match with --gres

# #cpu/#num_gpu_per_node
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1

{
    lr=1e-4

    # model_type="llama"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
    # model_type="llama_moe"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_16Select4-l2_norm
    # model_type="llama_moe"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-no-softmax/Clustering-l2-l2_norm/llama_13B-16Select4-gate_proj
    model_type="llama_moe"
    pretrained_model="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_model_bak/random_16select4_moe"

    tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
    # tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-no-softmax/Clustering-l2-l2_norm/llama_13B-16Select4-gate_proj
    dataset_dir=/mnt/petrelfs/share_data/quxiaoye/pretrain_LLAMA_all_data_processed

    per_device_train_batch_size=16
    per_device_eval_batch_size=1
    gradient_accumulation_steps=2
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
    output_dir=outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
    echo "output_dir: $output_dir"
    deepspeed_config_file=conf/deepspeed/bf16_zero1_default.json

    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIS ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "Node: $head_node"
    echo "Node IP: $head_node_ip"

    srun torchrun \
        --nnodes ${num_nodes} \
        --nproc_per_node ${num_gpu_per_node} \
        --node_rank $SLURM_NODEID \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        --rdzv_endpoint $head_node:29518 \
        smoe/entrypoint/cpt_fpt.py \
            --deepspeed ${deepspeed_config_file} \
            --model_name_or_path ${pretrained_model} \
            --model_type ${model_type} \
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
            --max_steps ${max_steps} \
            --max_train_samples 48828125 \
            --logging_strategy steps \
            --logging_steps 10 \
            --save_strategy steps \
            --save_total_limit 3 \
            --save_steps 1000 \
            --dataloader_num_workers 0 \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --block_size ${block_size} \
            --output_dir ${output_dir} \
            --overwrite_output_dir \
            --ddp_timeout 30000 \
            --logging_first_step True \
            --torch_dtype bfloat16 \
            --ddp_find_unused_parameters False \
            --gradient_checkpointing \
            --report_to none \
            --log_level info
}
#SBATCH --job-name=cpt-moe-fpt-test_lr_change
#改动前：--logging_steps 10 \
#改动后：--logging_steps 1 \
#改动前：没有--resume_from_checkpoint outputs/cpt-moe-fpt-test_lr_change-1700831/checkpoint-10
#改动后：有--resume_from_checkpoint outputs/cpt-moe-fpt-test_lr_change-1700831/checkpoint-10
#改动前：max_steps=$(echo "10^11 / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
#改动后：max_steps=20
#改动前：output_dir=outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
#改动后：output_dir=outputs/cpt-moe-fpt-test_lr_change-1700831
#改动前：--save_steps 1000 \
#改动后：--save_steps 10 \1
