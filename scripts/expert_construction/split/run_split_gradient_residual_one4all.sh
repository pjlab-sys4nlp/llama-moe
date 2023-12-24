#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"
echo ${llama_size}

expert_num_moe_list=(13 14 15)
expert_num_residual_list=(3 2 1)

#intermediate_size=8640 #  8640  11008  13824
#scale_factor=1
#expert_size=$(expr ${scale_factor} \* ${intermediate_size} / ${total_expert_num})

expert_size=864
# 540 1080 2160 4320 8640
# 688 1376 2752 5504 11008
# 864 1728 3456 6912 13824

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
for idx in "${!expert_num_moe_list[@]}"; do
  expert_num_moe=${expert_num_moe_list[${idx}]}
  expert_num_residual=${expert_num_residual_list[${idx}]}
  total_expert_num=$((${expert_num_moe} + ${expert_num_residual}))
  score_file_path=${data_path}/moefication_results/split/Gradients${total_expert_num}/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}
  visualization_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-neuron-overlap/cluster${total_expert_num}-${expert_num_residual}residual-${expert_num_moe}moe/${llama_size}-${expert_size}-${accumulate_level}-${importance_type}-${kernel}-${criterion}-${proj_type}

  for share_neurons in "True" "False"; do
    echo ${total_expert_num}\(${expert_num_moe}+${expert_num_residual}\) ${expert_size} ${share_neurons}

    OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
      python -m smoe.entrypoint.expert_construction.llama_split_gradient_residual \
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
      --criterion ${criterion} \
      --share_neurons ${share_neurons} &
    sleep 1.0
  done
done

chmod -R 755 ${save_path} >/dev/null 2>&1
