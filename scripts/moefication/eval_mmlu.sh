root_path=/mnt/petrelfs/ruanjiacheng/CODES/NLP/LLAMA2MOE/train-moe
tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
data_dir=/mnt/petrelfs/share_data/quxiaoye/llama_data/mmlu_data/
model_path=/mnt/petrelfs/share_data/quxiaoye/models
model_folder=llama_7B_MoE_16Select4-l2_norm
select_num=4


for i in '0' '1' '2' '3'
do
  OMP_NUM_THREADS=16 srun --partition=MoE --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1 \
    python ${root_path}/smoe/eval/eval_mmlu_moe_${i}.py \
    --data_dir ${data_dir} \
    --save_dir ${root_path}/outputs/${model_folder}-S${select_num}/ \
    --tokenizer_path ${tokenizer_path} \
    --model_path ${model_path}/${model_folder} \
    --select_num ${select_num} &
done

wait

OMP_NUM_THREADS=8 srun --partition=MoE --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1 \
  python ${root_path}/smoe/eval/gather_results.py \
  --txt_dir ${root_path}/outputs/${model_folder}-S${select_num}/ \
