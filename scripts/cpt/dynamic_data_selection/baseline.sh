#!/usr/bin/bash

#SBATCH --job-name=cpt-llama2_random_scale4_16gpus
#SBATCH --output=/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_16gpus/%x-%j.log
#SBATCH --error=/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_16gpus/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

# reserved spot

source ~/anaconda3/bin/activate smoe

{
    num_nodes=2        # should match with --nodes
    num_gpu_per_node=8 # should match with --gres

    # #cpu/#num_gpu_per_node
    export OMP_NUM_THREADS=32
    export LOGLEVEL=INFO
    #  export NCCL_DEBUG=INFO
    #  export TORCH_DISTRIBUTED_DEBUG=DETAIL
    #  export TORCH_SHOW_CPP_STACKTRACES=1
    #  export CUDA_LAUNCH_BLOCKING=1

    ##############################################################
    ############### LLAMA 7B Moefication 16Experts ###############
    #  comment="llama 7B residual, gradient, 2 + 2/14 | soft residual 2.0 | soft moe 2.0 | GPU num 1, per-device bs 64, lr 1e-4"
    #  pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEResidualForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-14Select2-2Residuals-688Neurons-Share

    ##############################################################
    ######## LLAMA 2 7B 16 Experts all kinds of ablations ########
    #  comment="llama 2 7B, residual 2, moefication gradient 2/14 | residual hard, moe soft 8.0 | GPU num 16, per-device bs 32, lr 3e-4"
    #  comment="llama 2 7B, residual 2, moefication gradient 2/14 | residual plain soft 8.0, moe soft 8.0 | GPU num 16, per-device bs 32, lr 3e-4"
    #  comment="llama 2 7B, residual 2, moefication gradient 2/14 | residual learn soft 8.0, moe soft 8.0 | GPU num 16, per-device bs 32, lr 3e-4"
    model_type="llama_moe"
    comment="llama 2 7B, random 4/16, per-device bsz 4M tokens, lr 1e-4, 16gpus"
    pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random/llama2_7B-16Select4-up_proj-Scale4.0

    #  comment="llama 2 7B, residual 2, share gradient 2/14 | residual hard, moe soft 8.0 | GPU num 16, per-device bs 32, lr 3e-4"
    #  comment="llama 2 7B, residual 2, share gradient 2/14 | residual plain soft 8.0, moe soft 8.0 | GPU num 16, per-device bs 32, lr 3e-4"
    #  comment="llama 2 7B, residual 2, share gradient 2/14 | residual learn soft 8.0, moe soft 8.0 | GPU num 16, per-device bs 32, lr 3e-4"
    #  comment="llama 2 7B, residual 2, share gradient 2/14 | residual learn soft 2.0, moe soft 2.0 | GPU num 16, per-device bs 32, lr 3e-4"
    #  pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEResidualForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama2_7B-14Select2-2Residuals-688Neurons-Share

    ##############################################################

    tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama2_7B
    dataset_dir=/mnt/petrelfs/share_data/quxiaoye/SlimPajama_processed
    validation_dir=/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized

    lr=1e-4
    final_lr_portion=0.1
    per_device_train_batch_size=8
    per_device_eval_batch_size=8
    gradient_accumulation_steps=8
    block_size=4096
    num_tokens="200*10^9"
    warmup_tokens="1*10^9"
    # warmup_tokens="0"
    eval_tokens="1*10^9"
    seed=1227
    deepspeed_config_file=conf/deepspeed/bf16_zero1_default.json

    num_selects=4

    max_steps=$(echo "${num_tokens} / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
    max_train_samples=$(echo "${num_tokens} / $block_size" | bc)
    echo "max_steps: $max_steps"
    echo "max_train_samples: $max_train_samples"
    global_bs=$(echo "$per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node" | bc)
    echo "global batch size: $global_bs"
    tokens_per_batch=$(echo "$global_bs * $block_size" | bc)
    echo "#tokens/batch: $tokens_per_batch"
    warmup_steps=$(echo "$warmup_tokens / $tokens_per_batch" | bc)
    echo "warmup tokens: $warmup_tokens, warmup steps: $warmup_steps"
    eval_steps=$(echo "$eval_tokens / $tokens_per_batch" | bc)
    echo "eval interval (tokens): $eval_tokens, steps: $eval_steps"

    data_cache=resources/cache
    base_dir="/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_16gpus"
    output_dir=$base_dir/outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
    mkdir -p $output_dir
    echo "output_dir: $output_dir"
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/env
    echo $comment > $output_dir/comment.txt
    echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    ln -snf $output_dir $base_dir/latest.dir
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "Node: $head_node"
    echo "Node IP: $head_node_ip"
    echo "Node list: $SLURM_JOB_NODELIS"

    srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29518 \
    smoe/entrypoint/cpt/cpt_fpt.py \
        --deepspeed ${deepspeed_config_file} \
        --model_name_or_path ${pretrained_model} \
        --model_type ${model_type} \
        --tokenizer_name_or_path ${tokenizer_path} \
        --dataset_dir ${dataset_dir} \
        --data_cache_dir ${data_cache} \
        --validation_dir ${validation_dir} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --do_train \
        --evaluation_strategy steps \
        --eval_steps ${eval_steps} \
        --seed ${seed} \
        --bf16 \
        --num_train_epochs 1 \
        --final_lr_portion ${final_lr_portion} \
        --optim adamw_torch \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate ${lr} \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --warmup_steps ${warmup_steps} \
        --max_steps ${max_steps} \
        --max_train_samples ${max_train_samples} \
        --save_strategy steps \
        --save_total_limit 1 \
        --save_steps ${eval_steps} \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory True \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --block_size ${block_size} \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --ddp_timeout 3600 \
        --ddp_find_unused_parameters False \
        --torch_dtype bfloat16 \
        --gradient_checkpointing \
        --logging_first_step True \
        --logging_strategy steps \
        --logging_steps 5 \
        --log_level info \
        --log_level_replica warning \
        --log_on_each_node False \
        --report_to none \
        --gate_type "TopKBalancedNoisyGate" \
        --calculator_type "UniversalCalculator" \
        --num_selects ${num_selects}
}
