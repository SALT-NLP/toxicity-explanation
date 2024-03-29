# MixGEN Model

### Training the MixGEN Model
Train the MixGEN model by calling train.py with a set of results from the ensembled models. You can use the pickle files generate by the testing files 
in the specialized knowledge model directories (src/seq2seq and src/knowledge). Here is an example command that ensembles the Expert Knowledge model 
trained on the target minority group classification, the explicit knowledge model using k=15 knowledge tuples and an implicit knowledge model trained on 
k=15 tuples generated by gpt2:

```
python train.py --model_type vanilla_ensemble --ensemble_files ../src/seq2seq/pred/bart_join_grp_checkpoint-3epoch_SBIC.v2.trn.pickle ../src/knowledge/pred/bart_knowledge_k_15_SBIC.v2.trn.pickle ../src/knowledge/pred/bart_implicit_stereotype_gpt2_k_15_SBIC.v2.trn.pickle --data_file ../../data/SBIC.v2.trn.csv
```

### Testing the MixGEN Model
Test the MixGEN model by calling test.py with prediction results on the test files and on the trained model. Here is an example command for the model 
trained above:

```
python test.py --model model/bart_vanilla_ensemble/ --model_type vanilla_ensemble --ensemble_files ../src/seq2seq/pred/bart_join_grp_checkpoint-3epoch_SBIC.v2.dev.pickle ../src/knowledge/pred/bart_knowledge_k_15_SBIC.v2.dev.pickle ../src/knowledge/pred/bart_implicit_stereotype_gpt2_k_15_SBIC.v2.dev.pickle --data_file ../../data/SBIC.v2.dev.csv
```

Remember to use the prediction files corresponding to the model prediction files used during training!
