#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_3B"
total_clusters=16

criterion=max                  #  min  max
kernel=l1_norm                 #  plain  l1_norm  l2_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
proj_type=up_proj              #  gate_proj  up_proj

if [ ${importance_type} = "feature_grad" ]; then
  template_postfix=grad
else
  template_postfix=change
fi

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
score_file_path=${data_path}/moefication_results/split/Gradients${total_clusters}/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}

gpus=0
cpus=4
for expert_size in 540 1080 2160 4320; do
  # 540 1080 2160 4320 8640
  # 688 1376 2752 5504 11008
  # 864 1728 3456 6912 13824
  save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-neuron-overlap/cluster${total_clusters}/${llama_size}-${expert_size}-${accumulate_level}-${importance_type}-${kernel}-${criterion}-${proj_type}

  OMP_NUM_THREADS=2 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    python -m smoe.entrypoint.visualization.visualize_expert_neuron_overlap \
    --model_path ${model_path} \
    --score_file_path ${score_file_path} \
    --save_path ${save_path} \
    --expert_size ${expert_size} \
    --score_file_template layers.{}.mlp.${proj_type}.weight.${template_postfix} \
    --criterion ${criterion} &
  sleep 0.7
done
