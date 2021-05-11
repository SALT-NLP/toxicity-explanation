import torch
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import AutoModelForCausalLM
from training_utils import *
from testing_utils import *

DATA_FILE = '../data/SBIC.v2.dev.csv'
MAX_LENGTH = 64

PRED_COL = ['HITId', 'post', 'sexYN', 'offensiveYN', 'intentYN', 'whoTarget', \
            'targetMinority','targetStereotype', 'speakerMinorityYN']

TEST_GPT_5EPOCH = {
                    'TO ACTUAL': 'pred/test/gpt_5epoch_dev_actual.csv',
                    'TO PRED': 'pred/test/gpt_5epoch_dev_pred.csv',
                    'TRAINED MODEL': 'model/gpt_5epoch/checkpoint-44734/',
                    'BASE MODEL': 'openai-gpt',
                    'SAMPLE SIZE': 2500
                  }

TEST_GPT2_5EPOCH = {
                    'TO ACTUAL': 'pred/test/gpt2_5epoch_dev_actual.csv',
                    'TO PRED': 'pred/test/gpt2_5epoch_dev_pred.csv',
                    'TRAINED MODEL': 'model/gpt2_5epoch/checkpoint-48150/',
                    'BASE MODEL': 'gpt2',
                    'SAMPLE SIZE': 2500
                  }

def generate_samples(df, model, tokenizer):
    i = 25
    j = 28
    posts = df.post.iloc[i:j].tolist()
    
    #inputs = tokenizer(posts, padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
    #print(inputs)
    #outputs = model.generate(**inputs, max_length=MAX_LENGTH*2)
    #print(outputs)

    for k in range(j - i):
      inputs = tokenizer(posts[k], max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
      outputs = model.generate(**inputs, max_length=MAX_LENGTH)
      print(outputs)

if __name__ == "__main__":
    active_test = TEST_GPT2_5EPOCH
    df = pd.read_csv(DATA_FILE)

    df = clean_post(df)
    tokenizer = setup_tokenizer(active_test['BASE MODEL'])
    model = AutoModelForCausalLM.from_pretrained(active_test['TRAINED MODEL'])
    model.eval()

    generate_samples(df, model, tokenizer)
