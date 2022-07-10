# Baselines

## Training the baseline models

There are two baseline models, GPT and GPT-2. Both can be trained using train.py. Hyperparameters are hard coded into a dictionary in the baseline file, 
but data files and dev files (optionally) can be passed in. If a dev file isn't passed in, the train file will be split (80/20). An example command to 
train a GPT-2 model on the Social Bias Frames dataset is given below:

```
python train.py --data_file ../data/SBIC.v2.trn.csv --model_type gpt2
```

To train on the implicit hate corpus, make sure to pass in the separator token as well as the impl_hate_data flag. You will encounter errors otherwise. 
Here's an example command:

```
python train.py --data_file ../data/implicit_hate_v1_stg3_posts.trn.tsv --sep '\t' --impl_hate_data --model_type gpt2
```

Once trained, the model will output a file: model/checkpoint-*. Make sure to rename this file to a more descriptive name!

## Testing the baseline models
To test the model, call test.py using the appropriate arguments. In general, you will pass in the model file and the saved model location. Here's an example using a saved gpt model trained on the Social Bias Frames dataset:

```
python test.py --model_type gpt --model_file model/gpt_baseline/ --data_file ../data/SBIC.v2.dev.csv
```

To test on the implicit hate dataset, you also pass the impl_hate_data flag and a separator token. For example, to test a gpt2 model trained on the 
Implicit Hate Dataset, you could use the following command:

```
python test.py --model_type gpt2 --model_file model/gpt2_implicit_hate/ --data_file ../data/implicit_hate_v1_stg3_posts.dev.tsv --sep '\t' --impl_hate_data
```
