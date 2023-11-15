#!/usr/bin/bash

#SBATCH --job-name=eval_ref_loss
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved

# reserved spot

source ~/anaconda3/bin/activate smoe

{
    num_nodes=1        # should match with --nodes
    num_gpu_per_node=1 # should match with --gres

    # #cpu/#num_gpu_per_node
    export OMP_NUM_THREADS=32
    export LOGLEVEL=INFO
    #  export NCCL_DEBUG=INFO
    #  export TORCH_DISTRIBUTED_DEBUG=DETAIL
    #  export TORCH_SHOW_CPP_STACKTRACES=1
    #  export CUDA_LAUNCH_BLOCKING=1

    ##############################################################
    model_type="llama"
    comment="llama 2 7B evaluation"
    # pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama2_7B
    pretrained_model=/mnt/petrelfs/zhutong/smoe/outputs/random_split_scale4_112gpus_11900steps_dense
    tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama2_7B
    ##############################################################

    dataset_dir=/mnt/petrelfs/share_data/quxiaoye/SlimPajama_processed
    validation_dir=/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized

    lr=2e-4
    final_lr_portion=0.1
    per_device_train_batch_size=8
    per_device_eval_batch_size=8
    block_size=4096
    seed=1227

    data_cache=resources/cache
    base_dir="."
    output_dir=$base_dir/outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
    mkdir -p $output_dir
    echo "output_dir: $output_dir"
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/env
    echo $comment > $output_dir/comment.txt
    # echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    # ln -snf $output_dir $base_dir/latest.dir
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "Node: $head_node"
    echo "Node IP: $head_node_ip"

    python smoe/entrypoint/cpt/cpt_fpt.py \
        --model_name_or_path ${pretrained_model} \
        --model_type ${model_type} \
        --tokenizer_name_or_path ${tokenizer_path} \
        --dataset_dir ${dataset_dir} \
        --data_cache_dir ${data_cache} \
        --validation_dir ${validation_dir} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --do_eval \
        --seed ${seed} \
        --bf16 \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory True \
        --block_size ${block_size} \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --ddp_timeout 3600 \
        --ddp_find_unused_parameters False \
        --torch_dtype bfloat16 \
        --logging_first_step True \
        --logging_strategy steps \
        --logging_steps 5 \
        --log_level info \
        --log_level_replica warning \
        --log_on_each_node False \
        --report_to none
}
