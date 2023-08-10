llama_size="llama_7B"
select_num=4
# moe_model_folder=Random-l2_norm/llama_7B-16Select4-up_proj
moe_model_folder='16select4_16card_bs16_checkpoint15000'


data_path=/mnt/petrelfs/share_data/quxiaoye

tokenizer_path=${data_path}/models/${llama_size}
data_dir=${data_path}/llama_data/mmlu_data/
model_path=${data_path}/continual_train_moe_models

gpus=1
cpus=$((gpus * 16))
for i in '0' '1' '2' '3'
do
  OMP_NUM_THREADS=16 srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpus} -n1 -c ${cpus} --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1 \
    python -m smoe.eval.eval_mmlu_moe_${i} \
    --data_dir ${data_dir} \
    --save_dir ${data_path}/eval_mmlu_outputs/${moe_model_folder}-S${select_num}/ \
    --tokenizer_path ${tokenizer_path} \
    --model_path ${model_path}/${moe_model_folder} \
    --select_num ${select_num} &
done

wait

chmod -R 777 ${data_path}/eval_mmlu_outputs/
