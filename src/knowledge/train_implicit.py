import sys
sys.path.append('../../shared/')

from datasets import Dataset,DatasetDict
from knowledge_utils import *
from knowledge import *
from utils import *
from transformers import BartForConditionalGeneration,AutoModelForCausalLM,AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.data.data_collator import DataCollatorForSeq2Seq

import os
import pickle
import random
import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--seed', type=int, default=685, help='Pass in a seed value. Default: 685')
    parser.add_argument('--batch_size', type=int, default=4, help='Pass in a batch size. Default: 4')
    parser.add_argument('--k', type=int, default=3, help='Pass in the number of implicit knowledge sentences to generate. Should be passed if using \'implicit_knowledge\' or \'concat\' model type. Default: 3')
    parser.add_argument('--sep', type=str, default=',', help='Pass in a separator for the data file.')
    parser.add_argument('--model_type', type=str, choices=['implicit_knowledge', 'target_minority', 'stereotype', 'concat'], required=True, help='Pass in a model type.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.trn.csv', help='Data File to load. Default: \'../../data/SBIC.v2.trn.csv\'')
    parser.add_argument('--dev_file', type=str, default=None, help='Data File to load. Default: \'../../data/SBIC.v2.trn.csv\'')
    parser.add_argument('--implicit_knowledge_generator', type=str, choices=['gpt2', 'openai-gpt'], help='Pick a generative model to generate implicit knowledge. Must be passed if using \'implicit_knowledge\' or \'concat\' model type Default: \'openai-gpt\'')
    parser.add_argument('--target_minority_model', type=str, help='Pass the path to an NLG model which generates target minorities. Must be passed if using \'implicit_knowledge\' or \'concat\' model type.')
    parser.add_argument('--implicit_knowledge_model', type=str, help='Pass the path to a trained NLG model which generates implicit knowledge. Must be passed if using \'stereotype\' model type.')
    parser.add_argument('--num_epochs', type=float, default=3.0, help='Pass in the number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Pass in the learning rate for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.5, help='Pass in the warmup ratio of the first epoch for training (only applies when training one epoch).')

    return parser.parse_args()

def check_args(args): 
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')

    if args.model_type == 'implicit_knowledge' or args.model_type == 'concat':
      if args.target_minority_model is None:
        raise ValueError('If training a bias generation model type, a path to a trained target minority generating model should be passed.')
      if args.implicit_knowledge_generator is None:
        raise ValueError('If training a bias generation model type, an implicit knowledge generator (gpt or gpt2) should be passed.')
    
      data_source = get_file_name(args.data_file)
      tm_model_name = get_file_name(args.target_minority_model)
      tm_pred_pickle_file = 'pred/' + tm_model_name + '_train.pickle'

      generation_pickle_file = 'data/implicit_generation_' + args.implicit_knowledge_generator + '_k_' + str(args.k) + '_' + data_source + '.pickle'
      print(generation_pickle_file)
      return args.target_minority_model, tm_pred_pickle_file, generation_pickle_file
    elif args.model_type == 'stereotype':
      if args.implicit_knowledge_model is None:
        raise ValueError('If training a stereotype generation model type, a path to a trained implicit knowledge generation should be passed.')
      return args.implicit_knowledge_model, None, None
    return SEQ2SEQ_MODEL_NAME, None, None

def process_df(args, data_file, tm_pred_pickle_file, generation_pickle_file):
    df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
    df = clean_post(df)

    if args.model_type == 'target_minority':
      df = clean_target(df, target_col='targetMinority')
    elif args.model_type == 'stereotype':
      df = clean_target(df, target_col='targetStereotype')
    elif args.model_type == 'implicit_knowledge':
      df = get_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file)
    elif args.model_type == 'concat':
      df = append_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file)

    dataset = Dataset.from_pandas(df)
    return dataset

if __name__ == '__main__':
    args = parse_args()
    print(args)
    model_load, tm_pred_pickle_file, generation_pickle_file = check_args(args)
    set_seed(args.seed)
    
    print('loading and tokenizing data ...')
    dataset = process_df(args, args.data_file, tm_pred_pickle_file, generation_pickle_file)
    if args.dev_file is not None:
      dev_dataset = process_df(args, args.dev_file, tm_pred_pickle_file, generation_pickle_file)
      datasets = DatasetDict({'train': dataset, 'test':dev_dataset})
    else:
      datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    print('tokenizing data ...')
    tokenizer, tokenized = tokenize_textgen_df(datasets, SEQ2SEQ_MODEL_NAME, padding=False, max_length=MAX_LENGTH)

    print('initializing model ...')
    model = BartForConditionalGeneration.from_pretrained(model_load)
    model.train()
    if torch.cuda.is_available():
      model.cuda()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True, max_length=MAX_LENGTH)
    train(
      model,
      tokenized,
      data_collator=data_collator,
      batch_size=args.batch_size,
      num_epochs=args.num_epochs,
      learning_rate=args.lr,
      warmup_ratio=args.warmup_ratio
    )
