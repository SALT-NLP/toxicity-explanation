import sys
sys.path.append('../shared/')

import torch
import argparse
import pandas as pd
import numpy as np
from gpt_utils import *
from utils import *
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, trainer_utils
import math

# Useful Constants
FROM_TRAIN_FILE = '../data/SBIC.v2.trn.csv'
FROM_DEV_FILE = '../data/SBIC.v2.dev.csv'
TO_TRAIN_FILE = 'data/baseline_train_text.csv'
TO_DEV_FILE = 'data/baseline_dev_text.csv'

WARMUP_DIV = 9.793
BLOCK_SIZE = 128

## Set all parameters here ##
GPT_DICT = {
    'MODEL_NAME':'openai-gpt',
    'LEARNING_RATE': 5e-6,
    'EPOCHS': 5.0,
    'SEED': 345
}

GPT2_DICT =  {
    'MODEL_NAME':'gpt2',
    'LEARNING_RATE': 1e-5,
    'EPOCHS': 5.0,
    'SEED': 434
}

def train(model, active_dict, lm_datasets):
  num_epochs = active_dict['EPOCHS']
  num_rows = lm_datasets['train'].num_rows
  batch_size = 4
  
  warmup_steps, save_steps, eval_steps = get_step_variables(
      num_rows,
      num_epochs,
      batch_size,
      warmup_div=WARMUP_DIV,
      warmup_one_epoch=False
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
      learning_rate = active_dict['LEARNING_RATE'],
      per_device_train_batch_size = batch_size,
      num_train_epochs = num_epochs,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=lm_datasets["train"],
      eval_dataset=lm_datasets["test"],
  )
  
  trainer.train()

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length % BLOCK_SIZE:
      remainder = BLOCK_SIZE - (total_length % BLOCK_SIZE)
    else:
      remainder = 0

    pad_extension = [tokenizer.pad_token_id for _ in range(remainder)]
    attention_extension = [0 for _ in range(remainder)]

    concatenated_examples['input_ids'].extend(pad_extension)
    concatenated_examples['attention_mask'].extend(attention_extension)
    total_length += remainder

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, help='Pass in a seed value. If nothing is passed a default is used.')
  parser.add_argument('--model_type', choices=['gpt','gpt2'], default='gpt', help='Pass either \'gpt\' or \'gpt2\'')
  return parser.parse_args()

def set_args(args):
  if args.model_type == 'gpt':
    active_dict = GPT_DICT
  else:
    active_dict = GPT2_DICT
  
  if args.seed is not None:
    active_dict['SEED'] = args.seed

  return active_dict

if __name__ == "__main__":
  args = parse_args()
  active_dict = set_args(args)
  print("Seed: ", active_dict['SEED'])
  
  print('cleaning and splitting dataset ...')
  dataset = clean_df(FROM_TRAIN_FILE, TO_TRAIN_FILE)
  datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
  
  # We need to create the model and tokenizer
  print('tokenizing and block grouping text ...')
  tokenizer = setup_tokenizer(active_dict['MODEL_NAME'])
  tokenized_datasets = datasets.map(lambda examples: tokenizer(examples["text"]), \
                                      batched=True, num_proc=4, \
                                      remove_columns=["text"])
  lm_datasets = tokenized_datasets.map(
      group_texts,
      batched=True,
      batch_size=1000,
      num_proc=4,
  )

  print('initializing model ...')
  trainer_utils.set_seed(active_dict['SEED'])
  model = AutoModelForCausalLM.from_pretrained(active_dict['MODEL_NAME'])
  model.resize_token_embeddings(len(tokenizer))
  model.train()
  
  train(model, active_dict, lm_datasets)
