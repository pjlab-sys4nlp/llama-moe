#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

expert_num_moe=15
expert_num_residual=1
total_expert_num=$((${expert_num_moe} + ${expert_num_residual}))

#intermediate_size=8640 #  8640  11008  13824
#scale_factor=1
#expert_size=$(expr ${scale_factor} \* ${intermediate_size} / ${total_expert_num})

expert_size=688
# 540 1080 2160 4320 8640
# 688 1376 2752 5504 11008
# 864 1728 3456 6912 13824

echo ${total_expert_num}\(${expert_num_moe}+${expert_num_residual}\) ${expert_size}

kernel=l1_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
criterion=max                  #  min  max
proj_type=up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
score_file_path=${data_path}/moefication_results/split/Gradients${total_expert_num}/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}
save_path=${data_path}/moefication_results/split
visualization_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-neuron-overlap/cluster${total_expert_num}-${expert_num_residual}residual-${expert_num_moe}moe/${llama_size}-${expert_size}-${accumulate_level}-${importance_type}-${kernel}-${criterion}-${proj_type}

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.moefication.llama_split_gradient_residual \
  --model_path ${model_path} \
  --score_file_path ${score_file_path} \
  --save_path ${save_path} \
  --visualization_path ${visualization_path} \
  --expert_num_moe ${expert_num_moe} \
  --expert_num_residual ${expert_num_residual} \
  --expert_size ${expert_size} \
  --template layers.{}.mlp.${proj_type}.weight \
  --kernel ${kernel} \
  --accumulate_level ${accumulate_level} \
  --importance_type ${importance_type} \
  --criterion ${criterion}

chmod -R 755 ${save_path} >/dev/null 2>&1
