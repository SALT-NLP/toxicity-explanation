#!/bin/bash

epochs=("3.0" "5.0")
lrs=("5e-4" "5e-5" "5e-6")

for epoch in ${epochs[@]}; do
  for lr in ${lrs[@]}; do
    echo "python train_implicit.py --batch_size 2 --sep '\t' --model_type stereotype --implicit_knowledge_model model/bart_implicit_bias_knowledge_gpt2_k_15/ --data_file ../../data/implicit_hate_v1_stg3_posts.trn.tsv --dev_file ../../data/implicit_hate_v1_stg3_posts.dev.tsv --num_epochs $epoch --lr $lr" >> grid_search_imp.txt
    python train_implicit.py --batch_size 2 --sep '\t' --model_type stereotype --implicit_knowledge_model model/bart_implicit_bias_knowledge_gpt2_k_15/ --data_file ../../data/implicit_hate_v1_stg3_posts.trn.tsv --dev_file ../../data/implicit_hate_v1_stg3_posts.dev.tsv --num_epochs $epoch --lr $lr
    model_file="$(ls ./model | grep "^checkpoint-*")"
    model_file_path="./model/$model_file"

    echo "python test_implicit.py --model $model_file_path --sep '\t' --model_type stereotype --data_file ../../data/implicit_hate_v1_stg3_posts.tst.tsv --use_cuda --generate_scores" >> grid_search_imp.txt
    python test_implicit.py --model $model_file_path --sep '\t' --model_type stereotype --data_file ../../data/implicit_hate_v1_stg3_posts.tst.tsv --use_cuda --generate_scores >> grid_search_imp.txt
    rm -rf pred/"$model_file"_implicit_hate_v1_stg3_posts.tst.pickle
  done
done
