#!/usr/bin/bash

#SBATCH --job-name=get-grad
#SBATCH --partition=MoE
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/train-moe/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/train-moe/logs/%x-%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --quotatype=spot

# reserved spot

num_nodes=1        # should match with --nodes
num_gpu_per_node=8 # should match with --gres

# #cpu/#num_gpu_per_node
export OMP_NUM_THREADS=2
export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_SHOW_CPP_STACKTRACES=1
#export CUDA_LAUNCH_BLOCKING=1

{
  ###################################################################
  #  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
  #  llama2_7B  llama2_13B  llama2_30B  llama2_base
  llama_size="llama2_13B"

  accumulate_level=sample        #  sample  total
  kernel=l1_norm                 #  plain  l1_norm  l2_norm
  importance_type=feature_change #  feature_grad  feature_change

  data_use_range_begin=0.0
  data_use_range_end=1.0

  data_path=/mnt/petrelfs/share_data/quxiaoye
  save_path=${data_path}/moefication_results/split
  pretrained_model=${data_path}/models/${llama_size}
  tokenizer_path=${data_path}/models/${llama_size}

  per_device_train_batch_size=1
  block_size=4096

  total_clusters=8 #  4  8  16  32
  dataset_dir=${data_path}/data/clustering_tokenized/${total_clusters}clusters
  #dataset_dir=/mnt/petrelfs/share_data/quxiaoye/test_tokenized.jsonl
  dataset_name=()
  for ((i = 0; i < ${total_clusters}; i++)); do
    dataset_name+=("${i}.jsonl")
  done

#  dataset_name=("0.jsonl")
#    dataset_name=("1.jsonl")
#    dataset_name=("2.jsonl")
#    dataset_name=("3.jsonl")
#    dataset_name=("4.jsonl")
#    dataset_name=("5.jsonl")
#    dataset_name=("6.jsonl")
    dataset_name=("7.jsonl")
#    dataset_name=("8.jsonl")
#    dataset_name=("9.jsonl")
#    dataset_name=("10.jsonl")
#    dataset_name=("11.jsonl")
#    dataset_name=("12.jsonl")
#    dataset_name=("13.jsonl")
#    dataset_name=("14.jsonl")
#    dataset_name=("15.jsonl")
#    dataset_name=("16.jsonl")
#    dataset_name=("17.jsonl")
#    dataset_name=("18.jsonl")
#    dataset_name=("19.jsonl")
#    dataset_name=("20.jsonl")
#    dataset_name=("21.jsonl")
#    dataset_name=("22.jsonl")
#    dataset_name=("23.jsonl")
#    dataset_name=("24.jsonl")
#    dataset_name=("25.jsonl")
#    dataset_name=("26.jsonl")
#    dataset_name=("27.jsonl")
#    dataset_name=("28.jsonl")
#    dataset_name=("29.jsonl")
#    dataset_name=("30.jsonl")
#    dataset_name=("31.jsonl")

  ###################################################################

  output_dir=/mnt/petrelfs/dongdaize.d/workspace/train-moe/outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
  echo "output_dir: $output_dir"

  deepspeed_config_file=/mnt/petrelfs/dongdaize.d/workspace/train-moe/conf/deepspeed/bf16.json

  nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  #port=$((6000 + $RANDOM))
  echo "Node: $head_node"
  echo "Node IP: $head_node_ip"

  for name in "${dataset_name[@]}"; do
    echo "@@@@@@@@@@@@@@@@@@@@@@" ${llama_size} ${name} "@@@@@@@@@@@@@@@@@@@@@@"
    srun torchrun \
      --nnodes ${num_nodes} \
      --nproc_per_node ${num_gpu_per_node} \
      --node_rank $SLURM_NODEID \
      --rdzv_id $RANDOM \
      --rdzv_backend c10d \
      --rdzv_endpoint $head_node:0 \
      smoe/entrypoint/moefication/llama_split_gradient_get_grads.py \
      --deepspeed ${deepspeed_config_file} \
      --model_name_or_path ${pretrained_model} \
      --tokenizer_name_or_path ${tokenizer_path} \
      --dataset_dir ${dataset_dir}/${name} \
      --validation_split_percentage 0 \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --do_train \
      --seed $RANDOM \
      --bf16 \
      --num_train_epochs 1 \
      --final_lr_portion 0.1 \
      --optim sgd \
      --learning_rate 0 \
      --weight_decay 0 \
      --max_grad_norm 1.0 \
      --warmup_steps 1000 \
      --logging_strategy steps \
      --logging_steps 10 \
      --save_strategy no \
      --dataloader_num_workers 8 \
      --block_size ${block_size} \
      --output_dir ${output_dir} \
      --overwrite_output_dir \
      --ddp_timeout 30000 \
      --logging_first_step True \
      --torch_dtype bfloat16 \
      --ddp_find_unused_parameters False \
      --report_to tensorboard \
      --gradient_checkpointing \
      --log_level info \
      --save_path ${save_path} \
      --total_clusters ${total_clusters} \
      --accumulate_level ${accumulate_level} \
      --kernel ${kernel} \
      --importance_type ${importance_type} \
      --data_use_range_begin ${data_use_range_begin} \
      --data_use_range_end ${data_use_range_end}
  done
}
chmod -R 755 ${save_path} >/dev/null 2>&1
