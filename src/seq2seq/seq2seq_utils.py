import sys
sys.path.append('../../shared/')

import pandas as pd
import numpy as np
import torch
import math
import pickle
from tqdm import tqdm
from datasets import Dataset
from transformers import BertTokenizer, BartTokenizer
from transformers import BartForConditionalGeneration
from seq2seq import *
from utils import *

TRAIN_DATA_FILE = '../../data/SBIC.v2.trn.csv'
DEV_DATA_FILE = '../../data/SBIC.v2.dev.csv'
TEST_DATA_FILE = '../../data/SBIC.v2.tst.csv'

################################ Data Preprocessing ################################

# Reads and cleans the CSV file passed in.
# Returns a dataset object.
def read_and_clean_csv(
    data_file,
    train=True,
    post_ids=None,
    sep=',',
):
    df = pd.read_csv(data_file, sep=sep, engine='python')
    #df = df[:100]
    df = clean_post(df)
    df = clean_target(df, train=train)
    
    if post_ids is not None:
      df = df[df.HITId.isin(post_ids)]
    
    dataset = Dataset.from_pandas(df)
    return dataset

################################ Tokenization ################################

# Tokenizes for BERT Classifier
def tokenize_classifier_df(
    dataset,
    classifier_tok_name,
):
    ## Local Function for tokenizing input.
    def tokenize(examples):
      classifier_tokenized = classifier_tok(
          examples['post'],
          padding="max_length",
          truncation=True,
          max_length=128,
      )
      return classifier_tokenized

    classifier_tok = BertTokenizer.from_pretrained(classifier_tok_name)
    tokenized = dataset.map(
        tokenize, batched=True,
        num_proc=1,
        remove_columns=['post','target','HITId']
    )
    return tokenized

# Takenizes a dataset, and a list of classifiers and computes the attentions
# for each classifier.
def get_classifier_attention(dataset, classifier_tok_name, classifiers, use_cuda=True):
    inputs = tokenize_classifier_df(dataset, classifier_tok_name)
    
    # Need to move to PyTorch tensor since huggingface tokenizer does
    # not always do it.
    input_ids = torch.tensor(inputs['input_ids'])
    attention_mask = torch.tensor(inputs['attention_mask'])
    if use_cuda and torch.cuda.is_available():
      input_ids = input_ids.cuda()
      attention_mask = attention_mask.cuda()
    
    batch_size = 20
    num_rows = input_ids.shape[0]
    num_batches = math.ceil(num_rows / batch_size)

    attentions = []
    for k,classifier in enumerate(classifiers):
      print("Running Classifier: ", classifier)
      model = BertForSequenceClassification.from_pretrained(classifier)
      if use_cuda and torch.cuda.is_available():
        model = model.cuda()
      
      attentions.append([])
      for batch in tqdm(range(num_batches)):
        i = batch * batch_size
        j = min(i + batch_size, num_rows)

        output = model(input_ids[i:j], attention_mask[i:j], output_attentions=True)
        attn_layers = output.attentions[-1].mean(dim=2)
        attentions[k].extend(attn_layers.tolist())
    attentions = np.stack(attentions, axis=1)
    return attentions


################################ Model Initialization ################################

# Model Initialization Function
def init_model(
    model_name,
    join=True,
    join_dropout=0.2,
    num_classifiers=4,
    num_classification_heads=12,
    train=True,
    use_cuda=True
):
    if join:
      model = BartForConditionalGenerationJoinModel.from_pretrained(
                  model_name,
                  join_dropout=join_dropout,
                  num_classifiers=num_classifiers,
                  num_classification_heads=num_classification_heads,
                  use_cuda=use_cuda
              )
    else:
      model = BartForConditionalGeneration.from_pretrained(model_name)
    
    if use_cuda and torch.cuda.is_available():
      model = model.cuda()
    
    if train:
      model.train()
    else:
      model.eval()
    
    return model

