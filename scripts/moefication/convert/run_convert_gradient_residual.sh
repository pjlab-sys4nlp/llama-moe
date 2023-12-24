#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"

share_neurons=True    #  True  False
num_experts=7          #  7  14  28
num_experts_residual=1 #  1  2  3  4
num_selects=1          #  1  2  3  4
expert_size=1376
# 540 1080 2160 4320 8640
# 688 1376 2752 5504 11008
# 864 1728 3456 6912 13824

score_scale_factor_residual=1.0 #  4.0  8.0  12.0  16.0
score_scale_factor=4.0          #  4.0  8.0  12.0  16.0

convert_type=LlamaMoEResidualForCausalLM #  LlamaMoEResidualModel  LlamaMoEResidualForCausalLM  LlamaMoEResidualForSequenceClassification

kernel=l1_norm
criterion=max                  #  min  max
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
proj_type=up_proj              #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/split/${llama_size}-Split-Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${num_experts}Experts-${num_experts_residual}Residuals-${expert_size}Neurons
save_path=${data_path}/models/${convert_type}/Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${llama_size}-${num_experts}Select${num_selects}-${num_experts_residual}Residuals-${expert_size}Neurons
if [ ${share_neurons} = "True" ]; then
  split_file_path=${split_file_path}-Share
  save_path=${save_path}-Share
fi

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.moefication.llama_convert_neuron_index_residual \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path "" \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts ${num_experts} \
  --num_experts_residual ${num_experts_residual} \
  --num_selects ${num_selects} \
  --score_scale_factor ${score_scale_factor} \
  --score_scale_factor_residual ${score_scale_factor_residual} \
  --convert_type ${convert_type} \
  --use_random_gate True
