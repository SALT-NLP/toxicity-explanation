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

