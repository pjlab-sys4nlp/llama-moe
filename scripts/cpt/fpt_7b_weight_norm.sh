#!/usr/bin/bash

#SBATCH --job-name=cpt-7b-16_16-gate-weight_norm
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
##SBATCH --output=logs/%x.log
##SBATCH --error=logs/%x.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
##SBATCH --time=5:00:00

source ~/anaconda3/bin/activate smoe

{
    num_nodes=2         # should match with --nodes
    num_gpu_per_node=4  # should match with --gres

    # #cpu/#num_gpu_per_node
    export OMP_NUM_THREADS=16
    export LOGLEVEL=INFO
    # export NCCL_DEBUG=INFO
    # export TORCH_DISTRIBUTED_DEBUG=DETAIL
    # export TORCH_SHOW_CPP_STACKTRACES=1
    # export CUDA_LAUNCH_BLOCKING=1
    # export ACCELERATE_DEBUG_MODE=1

    # comment="13B, expert 4/16, noisy gate, seq len 2048, lr=4e-4, expert weight re-scale"
    # comment="13B, expert 4/16, noisy gate, seq len 2048, lr=4e-4"
    # comment="llama2 7B, expert 16/16, noisy gate, seq len 4096, lr=3e-4, train weight norm and gate only, no balance, no noise"
    # comment="llama2 7B, expert 16/16, noisy gate, seq len 4096, lr=3e-4, train weight norm and gate only, no balance, no noise"
    # comment="llama2 7B, expert 4/16, noisy gate, seq len 4096, lr=3e-4, stage 2: gate and mlp only"
    comment="llama2 7B, moefication, expert 16/16, noisy gate without noise, seq len 4096, lr=1e-4, with weight norm (learnable scalar factor=4.0), bsz 4M, warmup 100M tokens, only train gate and weight norm"
    # comment="random initialized llama1-7B"
    # comment="random initialized llama1-13B"
    # comment="7B, expert 4/16, noisy gate, gradient shared neurons, w/o residual, w/o weight re-scale, lr2e-4"
    # comment="3B MoE, debug"

    # model_type="llama"
    # pretrained_model="/mnt/petrelfs/share_data/quxiaoye/models/llama_13B"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama1_7B_random
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama1_7B_random
    model_type="llama_moe"
    pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama2_7B-16Select4-688Neurons
    # pretrained_model=/mnt/petrelfs/zhutong/smoe/outputs/cpt-7b-4_16_noisygate-gate_stage1-2090437/checkpoint-4000
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random/llama_7B-16Select16-up_proj
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama2_7B-16Select4-688Neurons-Share
    # pretrained_model="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_3B-8Select2-4320Neurons-Share"
    # pretrained_model="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-688Neurons-Share"
    # pretrained_model="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-copy/Gradient-max-l1_norm-sample-feature_change/llama_13B-16Select4-864Neurons-Share"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_16Select4-l2_norm
    # pretrained_model="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-copy/Clustering-l2/llama_13B-16Select4-up_proj"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-no-softmax/Clustering-l2-l2_norm/llama_13B-16Select4-gate_proj
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2/llama_13B-16Select4-up_proj
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm/llama_13B-16Select4-up_proj
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random/llama_13B-16Select4-up_proj
    # pretrained_model=$1
    echo "==================> $pretrained_model <=================="

    tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama2_7B/
    # tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
    # tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-no-softmax-copy/Clustering-l2-l2_norm/llama_13B-16Select4-gate_proj
    # tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama1_7B_random
    # tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_13B
    # tokenizer_path="/mnt/petrelfs/share_data/quxiaoye/models/llama_3B"

    # dataset_dir=/mnt/petrelfs/share_data/quxiaoye/pretrain_LLAMA_all_data_processed
    dataset_dir=/mnt/petrelfs/share_data/quxiaoye/SlimPajama_processed/
    # dataset_dir=/mnt/petrelfs/zhutong/smoe/resources/slimpajama_samples_openllama3B_tokenized
    validation_dir=/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/

    lr=1e-4
    final_lr_portion=0.1
    per_device_train_batch_size=8
    per_device_eval_batch_size=8
    gradient_accumulation_steps=16
    num_tokens="10*10^9"
    warmup_tokens="0.1*10^9"
    # warmup_tokens="0"
    eval_tokens="5*10^9"
    seed=1227
    block_size=4096
    deepspeed_config_file=conf/deepspeed/bf16_zero1_default.json
    # deepspeed_config_file=conf/deepspeed/bf16_zero3.json

    max_steps=$(echo "${num_tokens} / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
    max_train_samples=$(echo "${num_tokens} / $block_size" | bc)
    echo "tokens: ${num_tokens}, warmup_tokens: ${warmup_tokens}, eval token interval: ${eval_tokens}"
    echo "max_steps: $max_steps, max_train_samples: $max_train_samples"
    global_bs=$(echo "$per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node" | bc)
    echo "global batch size: $global_bs"
    tokens_per_batch=$(echo "$global_bs * $block_size" | bc)
    echo "#tokens/batch: $tokens_per_batch"
    warmup_steps=$(echo "$warmup_tokens / $tokens_per_batch" | bc)
    echo "warmup tokens: $warmup_tokens, warmup steps: $warmup_steps"
    eval_steps=$(echo "$eval_tokens / $tokens_per_batch" | bc)
    echo "eval interval (tokens): $eval_tokens, steps: $eval_steps"

    data_cache=resources/cache
    output_dir=outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
    # output_dir=/mnt/petrelfs/share_data/quxiaoye/models/tzhu_model_bak/cpt-13b-16gpus-lr2e-4
    mkdir -p $output_dir
    echo "output_dir: $output_dir"
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/.env
    echo $comment > $output_dir/comment.txt

    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIS ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "Head Node: $head_node"
    echo "Head Node IP: $head_node_ip"
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
            --do_eval \
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
            --ddp_timeout 30000 \
            --ddp_find_unused_parameters False \
            --torch_dtype bfloat16 \
            --gradient_checkpointing \
            --logging_first_step True \
            --logging_strategy steps \
            --logging_steps 1 \
            --log_level info \
            --log_level_replica warning \
            --log_on_each_node False \
            --report_to none \
            --gate_type "TopKBalancedNoisyGate" \
            --calculator_type "UniversalCalculator" \
            --num_selects 16

}
