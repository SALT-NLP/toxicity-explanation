import sys
sys.path.append('../../shared/')

from datasets import Dataset, load_metric
from knowledge_utils import *
from knowledge import *
from utils import *
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from ast import literal_eval
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BartForConditionalGeneration
from transformers import Trainer, TrainingArguments

import nltk
import string
import os
import pickle
import pandas as pd
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='The path for the checkpoint folder',
    )
    
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers. Default: 16')
    parser.add_argument('--k', type=int, help='Pass in the number of implicit knowledge sentences to generate. Must be passed if using \'implicit_knowledge\' or \'concat\' model types.')
    parser.add_argument('--sep', type=str, default=',', help='Pass in a separator for the data file.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    parser.add_argument('--generate_scores', action='store_true', help='If True, will generate scores')
    parser.add_argument('--model_type', type=str, choices=['implicit_knowledge', 'target_minority', 'stereotype', 'concat'], required=True, help='Pass in a model type.')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--implicit_knowledge_generator', type=str, choices=['gpt2', 'openai-gpt'], help='Pick a generative model to generate implicit knowledge. Must be passed if using \'implicit_knowledge\' or \'concat\' model types.')
    parser.add_argument('--target_minority_model', type=str, help='Pass the path to an NLG model which generates target minorities. Must be passed if using \'implicit_knowledge\' or \'concat\' model types.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.dev.csv', help='Data File to load. Default: \'../../data/SBIC.v2.dev.csv\'')
    #parser.add_argument('--data_source', type=str, default='dev', choices=['dev','train','test', 'ext'], help='Data Source: Must be one of dev, train, test, or ext. Default: dev')
    parser.add_argument('--hitid_file', help='Path to HITID file for generation. (CURRENTLY NOT IN USE)')

    return parser.parse_args()

def check_args(args):
    model_path = args.model
    model_name = get_file_name(model_path)
    data_source = get_file_name(args.data_file)
    
    pickle_file = 'pred/' + model_name + '_' + data_source + '.pickle'
    results_file = 'results/' + model_name + '_' + data_source + '.csv'
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')

    if args.model_type == 'implicit_knowledge' or args.model_type == 'concat':
      if args.target_minority_model is None:
        raise ValueError('If testing a bias generation model type, a path to a trained target minority generating model should be passed.')
      if args.implicit_knowledge_generator is None:
        raise ValueError('If testing a bias generation model type, an implicit knowledge generator (gpt or gpt2) should be passed.')
      if args.k is None:
        raise ValueError('If testing a bias generation model type, k (an integer) should be passed.')
    
      tm_model_name = os.path.basename(os.path.normpath(args.target_minority_model))
      tm_pred_pickle_file = 'pred/' + tm_model_name + '_' + data_source + '.pickle'

      generation_pickle_file = 'data/implicit_generation_' + args.implicit_knowledge_generator + '_k_' + str(args.k) + '_' + data_source + '.pickle'
      return model_path, pickle_file, tm_pred_pickle_file, generation_pickle_file, results_file
    
    return model_path, pickle_file, None, None, results_file

if __name__ == '__main__':
    args = parse_args()
    model_path, pickle_file, tm_pred_pickle_file, generation_pickle_file, results_file = check_args(args)
    
    if not os.path.exists(pickle_file):
      print('loading and tokenizing data ...')
      df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
      df = clean_post(df)
      
      if args.model_type == 'stereotype':
        df = clean_target(df, target_col='targetStereotype', train=False)
      elif args.model_type == 'target_minority':
        df = clean_target(df, target_col='targetMinority', train=False)
      elif args.model_type == 'implicit_knowledge':
        df = get_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file, train=False)
      elif args.model_type == 'concat':
        df = append_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file, train=False)

      dataset = Dataset.from_pandas(df)
      seq2seq_tok, tokenized = tokenize_bart_df(dataset, SEQ2SEQ_MODEL_NAME, padding=False, train=False, max_length=MAX_LENGTH)
      
      print('initializing model ...')
      model = BartForConditionalGeneration.from_pretrained(model_path)
      model.eval()
      if args.use_cuda and torch.cuda.is_available():
        model.cuda()
      
      print('running model tests ...')
      torch_cols = ['input_ids', 'attention_mask']
      batch_iter = MinibatchIterator(tokenized, seq2seq_tok, batch_size=args.batch_size, torch_cols=torch_cols, use_cuda=args.use_cuda)
      generate_stereotypes(
          batch_iter,
          seq2seq_tok,
          model,
          model.get_encoder().forward,
          results_cols=['HITId','post','target'],
          pickle_file=pickle_file
      )
    
    if args.generate_scores or args.save_results_to_csv:
      print("generating base model scores ...")
      generate_scores(
        pickle_file,
        save_results_to_csv=args.save_results_to_csv,
        num_results=args.num_results,
        save_file=results_file
      )

