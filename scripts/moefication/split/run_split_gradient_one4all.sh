#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_3B"

intermediate_size=8640 #  8640  11008  13824

kernel=l1_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
criterion=max                  #  min  max
proj_type=up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/split

gpus=0
cpus=8

share_neurons=True
for expert_num in 16; do
  for expert_size in 864; do
    # 688 1376 2752 5504 11008
    # 864 1728 3456 6912 13824
    echo ${expert_num} ${expert_size} ${share_neurons}
    score_file_path=${data_path}/moefication_results/split/Gradients${expert_num}/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}

    OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
      python -m smoe.entrypoint.moefication.llama_split_gradient \
      --model_path ${model_path} \
      --score_file_path ${score_file_path} \
      --save_path ${save_path} \
      --expert_num ${expert_num} \
      --expert_size ${expert_size} \
      --template layers.{}.mlp.${proj_type}.weight \
      --kernel ${kernel} \
      --accumulate_level ${accumulate_level} \
      --importance_type ${importance_type} \
      --criterion ${criterion} \
      --share_neurons ${share_neurons} &
    sleep 1
  done
done

expert_num=16
scale_factor=1
share_neurons=False
expert_size=$(expr ${scale_factor} \* ${intermediate_size} / ${expert_num})
echo ${expert_num} ${expert_size} ${share_neurons}

OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.moefication.llama_split_gradient \
  --model_path ${model_path} \
  --score_file_path ${score_file_path} \
  --save_path ${save_path} \
  --expert_num ${expert_num} \
  --expert_size ${expert_size} \
  --template layers.{}.mlp.${proj_type}.weight \
  --kernel ${kernel} \
  --accumulate_level ${accumulate_level} \
  --importance_type ${importance_type} \
  --criterion ${criterion} \
  --share_neurons ${share_neurons}

wait
chmod -R 755 ${save_path} >/dev/null 2>&1
