#!/usr/bin/bash

set -vx

tokenizer_dir=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
data_dir=/mnt/petrelfs/zhutong/smoe/resources/clustering_samples_32
out_dir=/mnt/petrelfs/zhutong/smoe/resources/clustering_samples_32_tokenized
logs_dir=logs

mkdir -p $out_dir
mkdir -p $logs_dir

# for loop in: en_arxiv, en_book, en_c4, en_cc, en_stack, en_wikipedia, github
for data_type in $(ls $data_dir)
do
    log_path=logs/tokenize_${data_type}_32clusters.log
    nohup srun -p MoE -N1 -n1 --cpus-per-task=32 \
        python -m smoe.utils.tokenize \
            -f jsonl \
            -t $tokenizer_dir \
            -i $data_dir/$data_type \
            -o $out_dir/$data_type \
        1>${log_path} 2>&1 &
    echo "$data_type > $log_path"
done
