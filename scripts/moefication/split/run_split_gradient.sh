#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"
intermediate_size=11008
expert_num=16
expert_size=`expr 1 \* ${intermediate_size} / ${expert_num}`
echo ${expert_size}

kernel=l1_norm         #  plain  l1_norm  l2_norm
accumulate_level=total #  sample  total
criterion=max          #  min  max
share_neurons=False    #  True  False
proj_type=up_proj      #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
grad_file_path=${data_path}/moefication_results/split/Gradients/${llama_size}-Gradients-${kernel}-${accumulate_level}
save_path=${data_path}/moefication_results/split

gpus=0
cpus=8
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.moefication.llama_split_gradient \
  --model_path ${model_path} \
  --grad_file_path ${grad_file_path} \
  --save_path ${save_path} \
  --expert_size ${expert_size} \
  --template layers.{}.mlp.${proj_type}.weight \
  --kernel ${kernel} \
  --accumulate_level ${accumulate_level} \
  --criterion ${criterion} \
  --share_neurons ${share_neurons}

wait
chmod -R 777 ${save_path} >/dev/null 2>&1
