#!/usr/bin/bash

set -vx

tokenizer_dir=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
data_dir=/mnt/petrelfs/share_data/quxiaoye/pretrain_LLAMA_all_data
out_dir=/mnt/petrelfs/share_data/quxiaoye/pretrain_LLAMA_all_data_processed

# tokenizer_dir=/mnt/petrelfs/share_data/quxiaoye/models/llama_3B
# data_dir=/mnt/petrelfs/zhutong/smoe/resources/slimpajama_samples
# out_dir=/mnt/petrelfs/zhutong/smoe/resources/slimpajama_samples_openllama3B_tokenized

# tokenizer_dir=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
# data_dir=/mnt/petrelfs/zhutong/lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/val_set/final
# out_dir=/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized

logs_dir=logs

mkdir -p $logs_dir

# for loop in: en_arxiv, en_book, en_c4, en_cc, en_stack, en_wikipedia, github
for data_type in $(ls $data_dir)
do
    log_path=logs/tokenize_$data_type.log
    nohup srun -p MoE -N1 -n1 --cpus-per-task=32 \
        python -m smoe.utils.tokenize \
            -f jsonl \
            -t $tokenizer_dir \
            -i $data_dir/$data_type \
            -o $out_dir/$data_type \
        1>$logs_dir/tokenize_$data_type.log 2>&1 &
    echo "$data_type > $logs_dir/tokenize_$data_type.log"
done
