#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

intermediate_size=11008
retain_percent=99 #  99  98  95  90  80  75  70  60  50  40  30  25  20  13  10  06  05
expert_size=$((${retain_percent} * ${intermediate_size} / 100))
echo ${expert_size}

convert_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification
use_grad_sum=True                #  True  False

if [ ${use_grad_sum} = "True" ]; then
  expert_index=All
else
  expert_index=0
fi

criterion=min                  #  min  max
kernel=l1_norm                 #  plain  l1_norm  l2_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
proj_type=up_proj              #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/prune/${llama_size}-Prune-Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${expert_index}-0.${retain_percent}Percent-${expert_size}Neurons
save_path=${data_path}/models/${convert_type}-Prune/Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${llama_size}-${expert_index}-0.${retain_percent}Percent-${expert_size}Neurons

gpus=0
cpus=16
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=prune-convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.moefication.llama_convert_neuron_index \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path "" \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts 1 \
  --num_selects 1 \
  --convert_type ${convert_type} \
  --use_default_gate True

chmod -R 755 ${save_path} >/dev/null 2>&1
