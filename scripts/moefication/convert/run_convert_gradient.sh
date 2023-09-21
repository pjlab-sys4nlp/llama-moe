#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

num_experts=4          #  2  4  8  16
num_selects=1          #  1  2  4
score_scale_factor=1.0 #  1.0  16.0
expert_size=2752       #  688  1376  2752  5504  11008
#expert_size=1728                 #  864  1728  3456  6912  13824
share_neurons=True               #  True  False
convert_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification

kernel=l1_norm
criterion=max                  #  min  max
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
proj_type=up_proj              #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/split/${llama_size}-Split-Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${num_experts}Experts-${expert_size}Neurons
save_path=${data_path}/models/${convert_type}/Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${llama_size}-${num_experts}Select${num_selects}-${expert_size}Neurons

gpus=0
cpus=8
if [ ${share_neurons} = "True" ]; then
  split_file_path=${split_file_path}-Share
  save_path=${save_path}-Share
  OMP_NUM_THREADS=2 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    python -m smoe.entrypoint.moefication.llama_convert_neuron_index \
    --model_path ${model_path} \
    --split_file_path ${split_file_path} \
    --select_file_path "" \
    --save_path ${save_path} \
    --template layers.{}.mlp.${proj_type}.weight \
    --num_experts ${num_experts} \
    --num_selects ${num_selects} \
    --score_scale_factor ${score_scale_factor} \
    --convert_type ${convert_type} \
    --use_default_gate True
else
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    python -m smoe.entrypoint.moefication.llama_convert \
    --model_path ${model_path} \
    --split_file_path ${split_file_path} \
    --select_file_path "" \
    --save_path ${save_path} \
    --template layers.{}.mlp.${proj_type}.weight \
    --num_experts ${num_experts} \
    --num_selects ${num_selects} \
    --score_scale_factor ${score_scale_factor} \
    --convert_type ${convert_type} \
    --use_default_gate True
fi

chmod -R 755 ${save_path} >/dev/null 2>&1
