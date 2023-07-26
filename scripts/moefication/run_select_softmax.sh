llama_size="llama_7B" #  7B  13B  30B  base

num_experts=8
num_selects=2
select_criterion=l2_norm #  plain  positive  l2_norm

root_path=/mnt/petrelfs/dongdaize.d/workspace/llama-moe
model_path=/mnt/petrelfs/share_data/quxiaoye/models/${llama_size}
split_file_path=${root_path}/llama_moe_temp_files/${llama_size}-${num_experts}Expert-Split-Clustering
hidden_features_path=/mnt/petrelfs/share_data/quxiaoye/${llama_size}-Hidden-Features/
save_path=${root_path}/llama_moe_temp_files
template=layers.{}.mlp.gate_proj.weight

gpus=1
for specify_layer in "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "16 17 18 19" "20 21 22 23" "24 25 26 27" "28 29 30 31"; do # 并行启用任务
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=select --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c 24 --kill-on-bad-exit=1 \    -w SH-IDCA1404-10-140-54-${node} \
    python ${root_path}/run_moefication/llama_select_mlp.py \
    --model_path ${model_path} \
    --split_file_path ${split_file_path} \
    --hidden_features_path ${hidden_features_path} \
    --save_path ${save_path} \
    --template ${template} \
    --num_experts ${num_experts} \
    --num_selects ${num_selects} \
    --select_criterion ${select_criterion} \
    --specify_layer ${specify_layer} \
    --use_softmax & # 并行运行下一命令
done
