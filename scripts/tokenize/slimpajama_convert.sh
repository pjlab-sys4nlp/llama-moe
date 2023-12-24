#!/usr/bin/bash

# set -vx

content_column=input_ids
src_tokenizer_dir=/mnt/petrelfs/share_data/zhutong/models/llama2_7B
tokenizer_dir=/mnt/petrelfs/share_data/zhutong/models/Mistral-7B-v0.1

data_dir=/mnt/petrelfs/share_data/zhutong/data/slimpajama_fluency_llama_middle_parts
out_dir=/mnt/petrelfs/share_data/zhutong/data/slimpajama_fluency_mistral_middle_parts
# data_dir=/mnt/petrelfs/share_data/zhutong/data/llama1_7B_val_set_tokenized
# out_dir=/mnt/petrelfs/share_data/zhutong/data/mixtral_val_set_tokenized


logs_dir=logs

mkdir -p $logs_dir

# for loop in: en_arxiv, en_book, en_c4, en_cc, en_stack, en_wikipedia, github
# for data_type in $(ls $data_dir)
for data_type in "en_arxiv" "en_book" "en_c4" "en_stack" "en_wikipedia" "github"
do
    # get all parts from source data dir
    for part in $(ls $data_dir/$data_type)
    do
        echo "tokenizing $data_dir/$data_type/$part - $(ls $data_dir/$data_type/$part | wc -l)"
        log_path=logs/tokenize-$data_type-$part.log
        nohup srun -p MoE_T -N1 -n1 --cpus-per-task=32 \
            python -m smoe.utils.tokenize \
                -f jsonl \
                -c $content_column \
                -s $src_tokenizer_dir \
                -t $tokenizer_dir \
                -i $data_dir/$data_type/$part \
                -o $out_dir/$data_type/$part \
            1>$log_path 2>&1 &
        # echo "$data_type/$part > $log_path"
        sleep 3
    done

    # log_path=logs/tokenize_$data_type.log
    # nohup srun -p MoE_T -N1 -n1 --cpus-per-task=32 \
    #     python -m smoe.utils.tokenize \
    #         -f jsonl \
    #         -s $src_tokenizer_dir \
    #         -c $content_column \
    #         -t $tokenizer_dir \
    #         -i $data_dir/$data_type \
    #         -o $out_dir/$data_type \
    #     1>$logs_dir/tokenize_$data_type.log 2>&1 &
    # echo "$data_type > $logs_dir/tokenize_$data_type.log"
done
