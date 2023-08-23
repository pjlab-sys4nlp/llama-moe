#!/usr/bin/bash

#SBATCH --job-name=split-grad
#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --gres=gpu:8

num_nodes=1        # should match with --nodes
num_gpu_per_node=8 # should match with --gres

# #cpu/#num_gpu_per_node
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO

###################################################################
#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

model_intermediate_dize=11008 #  11008
expert_num=16                 #  8  16
expert_size=$((${model_intermediate_dize} / ${expert_num}))

proj_type=gate_proj     #  gate_proj  up_proj
accumulate_level=sample #  sample  total
kernel=l1_norm          #  l1_norm  l2_norm

data_use_range_begin=0.0
data_use_range_end=1.0

data_path=/mnt/petrelfs/share_data/quxiaoye
save_path=${data_path}/moefication_results/split
pretrained_model=${data_path}/models/${llama_size}
tokenizer_path=${data_path}/models/${llama_size}
dataset_dir=${data_path}/pretrain_LLAMA_all_data_processed
###################################################################

per_device_train_batch_size=16
block_size=2048

output_dir=./outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
echo "output_dir: $output_dir"

deepspeed_config_file=conf/deepspeed/bf16.json

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
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
  --rdzv_endpoint $head_node:29999 \
  smoe/entrypoint/cpt_fpt.py \
  --deepspeed ${deepspeed_config_file} \
  --model_name_or_path ${pretrained_model} \
  --tokenizer_name_or_path ${tokenizer_path} \
  --dataset_dir ${dataset_dir} \
  --validation_split_percentage 0.001 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --do_train \
  --seed $RANDOM \
  --bf16 \
  --num_train_epochs 1 \
  --final_lr_portion 0.1 \
  --optim adamw_torch \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --learning_rate 0 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_steps 2000 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy steps \
  --save_total_limit 3 \
  --save_steps 1000 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 1 \
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
  --expert_size ${expert_size} \
  --template layers.{}.mlp.${proj_type}.weight \
  --accumulate_level ${accumulate_level} \
  --kernel ${kernel} \
  --data_use_range_begin ${data_use_range_begin} \
  --data_use_range_end ${data_use_range_end}
