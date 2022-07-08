import sys
sys.path.append('../shared/')

import nltk
import torch
import argparse
import tqdm
import pandas as pd
import numpy as np
from gpt_utils import *
from utils import *
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, trainer_utils
from datasets import DatasetDict
import math

# Useful Constants
FROM_TRAIN_FILE = '../data/SBIC.v2.trn.csv'
FROM_DEV_FILE = '../data/SBIC.v2.dev.csv'
#TO_TRAIN_FILE = 'data/baseline_train_text.csv'
#TO_DEV_FILE = 'data/baseline_dev_text.csv'

WARMUP_DIV = 9.793
BLOCK_SIZE = 128

## Set all parameters here ##
#GPT_DICT = {
#    'MODEL_NAME':'openai-gpt',
#    'LEARNING_RATE': 5e-6,
#    'EPOCHS': 5.0,
#    'SEED': 345
#}
#
#GPT2_DICT =  {
#    'MODEL_NAME':'gpt2',
#    'LEARNING_RATE': 1e-5,
#    'EPOCHS': 5.0,
#    'SEED': 434
#}

class GPT2Dataset(Dataset):
  def __init__(self, data, tokenizer, gpt2_type="gpt2", max_length=90):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in tqdm(data):
      encodings_dict = tokenizer(txt['text'], truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]

def train(model, args, lm_datasets):
  num_rows = lm_datasets['train'].num_rows
  
  warmup_steps, save_steps, eval_steps = get_step_variables(
      num_rows,
      args.num_epochs,
      args.batch_size,
      warmup_div=WARMUP_DIV,
      warmup_one_epoch=True
  )
  print("Linear Warm Up: ", warmup_steps)
  print("Save Steps: ", save_steps)
  print("Eval Steps: ", eval_steps)
  
  training_args = TrainingArguments(
      output_dir = 'model',
      evaluation_strategy = 'steps',
      eval_steps = eval_steps,
      save_steps = save_steps,
      save_total_limit = 1,
      warmup_steps = warmup_steps,
      learning_rate = args.lr,
      per_device_train_batch_size = args.batch_size,
      num_train_epochs = args.num_epochs,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=lm_datasets["train"],
      eval_dataset=lm_datasets["test"],
  )
  
  trainer.train()

def set_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', default=4853, type=int, help='Pass in a seed value.')
  parser.add_argument('--batch_size', default=2, type=int, help='Pass in a batch size.')
  parser.add_argument('--num_epochs', default=5.0, type=float, help='Pass in the number of training epochs.')
  parser.add_argument('--lr', default=5e-5, type=float, help='Pass in the learning rate for training.')
  parser.add_argument('--model_name', choices=['openai-gpt','gpt2'], default='gpt2', help='Pass either \'openai-gpt\' or \'gpt2\'.')
  parser.add_argument('--data_file', default='../data/implicit_hate_v1_stg3_posts.trn.tsv', help='Data file for training.')
  parser.add_argument('--dev_file', default='../data/implicit_hate_v1_stg3_posts.dev.tsv', help='Dev file for training.')
  parser.add_argument('--sep', default=',', help='Separator for file read.')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  model_name = ''
  print(args)
  print()
  
  print('cleaning and splitting dataset ...')
  print()
  trn_dataset = clean_df(args.data_file, sep=args.sep, impl=True)
  dev_dataset = clean_df(args.dev_file, sep=args.sep, impl=True)
  #datasets = DatasetDict({'train': dataset, 'test': dev_dataset})
  
  max_len_np = 128

  # We need to create the model and tokenizer
  print('tokenizing and block grouping text ...')
  print()
  
  tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token='[STR]', eos_token='[END]', sep_token='[SEP]', pad_token='[PAD]')
  print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
  print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
  print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
  print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))
  print()

  trn_dataset = GPT2Dataset(trn_dataset, tokenizer, max_length=max_len_np)
  dev_dataset = GPT2Dataset(dev_dataset, tokenizer, max_length=max_len_np)
  exit()

  tokenize_func = lambda examples: tokenizer(
                                      examples["text"],
                                      padding='max_length',
                                      truncation=True,
                                      max_length=BLOCK_SIZE
                                   )
  tokenized_datasets = datasets.map(
                          tokenize_func,
                          batched=True, num_proc=4,
                          remove_columns=["text"]
                       )

  lm_datasets = tokenized_datasets.map(
      set_labels,
      batched=True,
      batch_size=1000,
      num_proc=4,
  )
  
  print('initializing model ...')
  trainer_utils.set_seed(args.seed)
  model = AutoModelForCausalLM.from_pretrained(args.model_name)
  model.resize_token_embeddings(len(tokenizer))
  model.train()
  
  train(model, args, lm_datasets)
