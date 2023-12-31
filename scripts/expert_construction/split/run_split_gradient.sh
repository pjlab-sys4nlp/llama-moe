#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_3B"

share_neurons=True #  True  False
expert_num=4

#intermediate_size=8640 #  8640  11008  13824
#scale_factor=4
#expert_size=$(expr ${scale_factor} \* ${intermediate_size} / ${expert_num})

expert_size=8640
# 540 1080 2160 4320 8640
# 688 1376 2752 5504 11008
# 864 1728 3456 6912 13824

echo ${expert_num} ${expert_size} ${share_neurons}

kernel=l1_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
criterion=max                  #  min  max
proj_type=up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
score_file_path=${data_path}/moefication_results/split/Gradients${expert_num}/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}
save_path=${data_path}/moefication_results/split
visualization_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-neuron-overlap/cluster${expert_num}/${llama_size}-${expert_size}-${accumulate_level}-${importance_type}-${kernel}-${criterion}-${proj_type}

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.expert_construction.llama_split_gradient \
  --model_path ${model_path} \
  --score_file_path ${score_file_path} \
  --save_path ${save_path} \
  --visualization_path ${visualization_path} \
  --expert_num ${expert_num} \
  --expert_size ${expert_size} \
  --template layers.{}.mlp.${proj_type}.weight \
  --kernel ${kernel} \
  --accumulate_level ${accumulate_level} \
  --importance_type ${importance_type} \
  --criterion ${criterion} \
  --share_neurons ${share_neurons}

chmod -R 755 ${save_path} >/dev/null 2>&1
