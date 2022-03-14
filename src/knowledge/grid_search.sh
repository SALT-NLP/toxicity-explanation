#!/bin/bash

epochs=("3.0" "5.0")
lrs=("5e-4" "5e-5" "5e-6")

for epoch in ${epochs[@]}; do
  for lr in ${lrs[@]}; do
    echo "python train.py --k 20 --batch_size 2 --data_file ../../data/implicit_hate_v1_stg3_posts.trn.tsv --dev_file ../../data/implicit_hate_v1_stg3_posts.dev.tsv --model_type input --sep '\t' --num_epochs $epoch --lr $lr" >> grid_search.txt
    python train.py --k 20 --batch_size 2 --data_file ../../data/implicit_hate_v1_stg3_posts.trn.tsv --dev_file ../../data/implicit_hate_v1_stg3_posts.dev.tsv --model_type input --sep '\t' --num_epochs $epoch --lr $lr
    model_file="$(ls ./model | grep "^checkpoint-*")"
    model_file_path="./model/$model_file"

    echo "python test.py --model $model_file_path --k 20 --sep '\t' --data_file ../../data/implicit_hate_v1_stg3_posts.tst.tsv --model_type input --use_cuda --generate_scores" >> grid_search.txt
    python test.py --model $model_file_path --k 20 --sep '\t' --data_file ../../data/implicit_hate_v1_stg3_posts.tst.tsv --model_type input --use_cuda --generate_scores >> grid_search.txt
    rm -rf pred/"$model_file"_implicit_hate_v1_stg3_posts.tst.pickle
  done
done
