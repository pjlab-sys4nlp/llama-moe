#!/usr/bin/bash

llama_size="llama_7B" #  7B  13B  30B  base
num_experts=8
num_selects=2

split_type=Clustering-l2                #  Clustering-l2  Clustering-cos
select_type=l2_norm                     #  plain  positive  l2_norm
convert_type=LlamaMoEForCausalLM        #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification
template=layers.{}.mlp.gate_proj.weight #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-${split_type}
select_file_path=${data_path}/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
save_path=${data_path}/models/${convert_type}/${llama_size}_${num_experts}Select${num_selects}-${select_type}

gpus=1
cpus=16
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.moefication.llama_convert \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path ${select_file_path} \
  --save_path ${save_path} \
  --template ${template} \
  --num_experts ${num_experts} \
  --num_selects ${num_selects} \
  --convert_type ${convert_type}
