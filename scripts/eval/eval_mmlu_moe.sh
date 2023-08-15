#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"

num_experts=8       #  8  16
num_selects=2       #  2  4
split_type=Clustering-cos   #  Graph-l1_norm  Graph-l2_norm  Clustering-l2  Clustering-cos  Random
select_type=l2_norm #  plain  positive  l1_norm  l2_norm
proj_type=up_proj   #  gate_proj  up_proj

set_num_selects=2

data_path=/mnt/petrelfs/share_data/quxiaoye
tokenizer_path=${data_path}/models/${llama_size}
data_dir=${data_path}/llama_data/mmlu_data/
model_path=${data_path}/models/LlamaMoEForCausalLM/${split_type}-${select_type}/${llama_size}_${num_experts}Select${num_selects}-${proj_type}
save_path=${data_path}/eval_mmlu_outputs/${split_type}-${select_type}/${llama_size}_${num_experts}Select${num_selects}-${proj_type}-S${set_num_selects}

# model_path=${data_path}/models/llama_7B-16Select4-up_proj
# save_path=${data_path}/eval_mmlu_outputs/16select4_16card_bs16_checkpoint15000

gpus=1
cpus=$((gpus * 16))
for i in '0' '1' '2' '3'; do
  OMP_NUM_THREADS=16 srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpus} -n1 -c ${cpus} --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.eval.eval_mmlu_moe_${i} \
    --data_dir ${data_dir} \
    --save_dir ${save_path} \
    --tokenizer_path ${tokenizer_path} \
    --model_path ${model_path} \
    --select_num ${set_num_selects} &
  sleep 0.5s
done

wait
chmod -R 777 ${save_path}
