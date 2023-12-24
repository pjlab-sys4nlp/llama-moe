#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

retain_percent=0.20 #  0.99 0.98 0.95 0.90 0.80 0.75 0.70 0.60 0.50 0.40 0.30 0.25 0.20 0.13 0.10 0.06 0.05
use_grad_sum=True   #  True  False

if [ ${use_grad_sum} = "True" ]; then
  expert_index=All
else
  expert_index=0
fi

criterion=max                  #  min  max
kernel=l1_norm                 #  plain  l1_norm  l2_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
proj_type=up_proj              #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
grad_file_path=${data_path}/moefication_results/split/Gradients/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}
save_path=${data_path}/moefication_results/prune

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.expert_construction.llama_prune_gradient \
  --model_path ${model_path} \
  --grad_file_path ${grad_file_path} \
  --save_path ${save_path} \
  --retain_percent ${retain_percent} \
  --expert_index ${expert_index} \
  --template layers.{}.mlp.${proj_type}.weight \
  --kernel ${kernel} \
  --accumulate_level ${accumulate_level} \
  --importance_type ${importance_type} \
  --criterion ${criterion} \
  --use_grad_sum ${use_grad_sum}

chmod -R 755 ${save_path} >/dev/null 2>&1
