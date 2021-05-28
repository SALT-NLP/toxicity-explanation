import sys
sys.path.append('../shared/')

import pandas as pd
import numpy as np
import os
import math
import pickle
import argparse
from seq2seq_utils import *
from seq2seq import BartForConditionalGenerationJoinModel
from torch import nn, torch
from datasets import Dataset,load_metric
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from train import *
from utils import *

# Useful constants
CLASSIFIER_TOK_NAME = 'bert-base-uncased'
SEQ2SEQ_TOK_NAME = 'facebook/bart-base'

FROM_DATA_FILE = '../../data/SBIC.v2.dev.csv'
TO_DATA_FILE = 'data/clean_dev_df.csv'
HITID_SET = {
  '30QQTY5GMKEKBSMET2NHS1NAUF57UX', '30QQTY5GMKEKBSMET2NHS1NAUH7U7Q', '30U1YOGZGAQKDOVKVAV3DSFIVX7SDN',
  '30U1YOGZGAQKDOVKVAV3DSFIVZTDSY', '30U1YOGZGAQKDOVKVAV3DSFIX1KSDA', '30U1YOGZGAQKDOVKVAV3DSFIY4VDSD',
  '30UZJB2POH6LPUVCQPCJ78JEQ0B531', '30UZJB2POH6LPUVCQPCJ78JEQVI53Y', '30UZJB2POH6LPUVCQPCJ78JET82359',
  '30Y6N4AHYPQ8C9V7GLVYNIAM9OXDRF', '30Y6N4AHYPQ8C9V7GLVYNIAMA7ERDD', '30Y6N4AHYPQ8C9V7GLVYNIAMC3UDR9',
  '30Y6N4AHYPQ8C9V7GLVYNIAMC3VRDO', '30Z7M1Q8UYE4WXDZX2YW607B198A8T', '30Z7M1Q8UYE4WXDZX2YW607BYWI8A8',
  '311HQEI8RSA1XRGOZPMP9T2PW157ZF', '311HQEI8RSA1XRGOZPMP9T2PWWBZ73', '311HQEI8RSA1XRGOZPMP9T2PWXJZ7D',
  '311HQEI8RSA1XRGOZPMP9T2PWXK7ZM', '311HQEI8RSA1XRGOZPMP9T2PWZGZ7E', '311HQEI8RSA1XRGOZPMP9T2PWZN7ZT',
  '311HQEI8RSA1XRGOZPMP9T2PY5HZ7T', '311HQEI8RSA1XRGOZPMP9T2PY5I7Z2', '311HQEI8RSA1XRGOZPMP9T2PYAW7ZQ',
  '311HQEI8RSA1XRGOZPMP9T2PZ9D7Z6', '3126F2F5F8XSS2TSZO2TO5SS8J8EPH', '3126F2F5F8XSS2TSZO2TO5SS908EPG',
  '3126F2F5F8XSS2TSZO2TO5SSATEPEK', '31ANT7FQN8W0J22B5A1LB2KO9005H5', '31ANT7FQN8W0J22B5A1LB2KOCEWH58'
}

def generate_stereotypes(tokenized, seq2seq_tok, model, batch_size=16, use_cuda=True):
    results = [[],[],[]]
    num_batches = math.ceil(tokenized.num_rows / batch_size)

    for batch in tqdm(range(num_batches)):
      i = batch * batch_size
      j = min(tokenized.num_rows, i + batch_size)
      
      _, output_strs = generate_batch(tokenized, seq2seq_tok, model, i, j, use_cuda=use_cuda)
      results[0].extend(tokenized['target'][i:j])
      results[1].extend(output_strs)
      results[2].extend(tokenized['HITId'][i:j])
    return results

def print_example_outputs(tokenized, tokenizer, model, use_cuda=True):
    input_strs, output_strs = generate_batch(tokenized, tokenizer, model, 120, 130, use_cuda=use_cuda)
    for i in range(len(input_strs)):
      print('Input Sentence: ', input_strs[i])
      print('Output Stereotype: ', output_strs[i])
      print('\n')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m',
        '--models',
        nargs='+',
        required=True,
        help='The path for the checkpoint folder. You can pass multiple.',
    )
    parser.add_argument(
        '-c',
        '--classifiers',
        nargs='+',
        required=True,
        help='The path for the classifier. You can pass multiple, corresponding to the models above.',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse Args
    args = parse_args()
    model_path_0 = args.models[0]
    model_path_1 = args.models[1]
    join = False
    
    if join and args.classifiers is None:
      raise ValueError('You have selected a join model, but have not provided any classifiers as args.')
    
    # Tokenize Data
    print("cleaning csv ...")
    dataset = read_and_clean_csv(FROM_DATA_FILE, TO_DATA_FILE, train=False)
    dataset = dataset.filter(in_hitid_set)

    num_classifiers = 0
    num_classification_heads = 0
    
    if join:
      print("tokenizing and classifying data ...")
      attentions_0 = get_classifier_attention(dataset, CLASSIFIER_TOK_NAME, [args.classifiers[0]], args.use_cuda)
      attentions_1 = get_classifier_attention(dataset, CLASSIFIER_TOK_NAME, [args.classifiers[1]], args.use_cuda)
      
      print("mapping classifier attention to dataset ...")
      dataset_0 = map_column_to_dataset(dataset, attentions_0, 'classifier_attention')
      dataset_1 = map_column_to_dataset(dataset, attentions_1, 'classifier_attention')
      
      num_classifiers = attentions_0.shape[1]
      num_classification_heads = attentions_0.shape[2]
      
      print('tokenizing data for bart ...')
      seq2seq_tok, dataset_0 = tokenize_bart_df(
          dataset_0,
          SEQ2SEQ_TOK_NAME,
          train=False
      )

      seq2seq_tok, dataset_1 = tokenize_bart_df(
          dataset_1,
          SEQ2SEQ_TOK_NAME,
          train=False
      )
      
      # Initialize Model
      print('initializing model ...')
      model_0 = init_model(
          model_path_0,
          join=join,
          num_classifiers=num_classifiers,
          num_classification_heads=num_classification_heads,
          train=False,
          use_cuda=args.use_cuda
      )
      model_1 = init_model(
          model_path_1,
          join=join,
          num_classifiers=num_classifiers,
          num_classification_heads=num_classification_heads,
          train=False,
          use_cuda=args.use_cuda
      )
      
      # Run Tests
      #i = 145
      #j = 149

      #print('Model 1')
      #_, output_strs_0, encoder_outputs_0 = generate_batch(dataset_0, seq2seq_tok, model, i, j, use_cuda=args.use_cuda)
      #print('Model 2')
      #_, output_strs_1, encoder_outputs_1 = generate_batch(dataset_1, seq2seq_tok, model, i, j, use_cuda=args.use_cuda)
      
      #input_ids_0 = torch.tensor(dataset_0['input_ids'][i:j])
      #attention_mask_0 = torch.tensor(dataset_0['attention_mask'][i:j])
      #classifier_attention_0 = torch.tensor(dataset_0['classifier_attention'][i:j])
      #
      #input_ids_1 = torch.tensor(dataset_1['input_ids'][i:j])
      #attention_mask_1 = torch.tensor(dataset_1['attention_mask'][i:j])
      #classifier_attention_1 = torch.tensor(dataset_1['classifier_attention'][i:j])

      #output_0 = model.forward(input_ids_0, attention_mask_0, encoder_outputs = encoder_outputs_0)
      #output_1 = model.forward(input_ids_1, attention_mask_1, encoder_outputs = encoder_outputs_1)
      #print(output_0.logits == output_1.logits)
      
      #print(output_strs_0)
      #print(output_strs_1)
      
      print('running model tests ...')
      results_0 = generate_stereotypes(
        dataset_0,
        seq2seq_tok,
        model_0,
        batch_size=args.batch_size,
        use_cuda=args.use_cuda
      )
      
      results_1 = generate_stereotypes(
        dataset_1,
        seq2seq_tok,
        model_1,
        batch_size=args.batch_size,
        use_cuda=args.use_cuda
      )
    
    #num_diff = 0
    #hitid_diff = []
    for i in range(len(results_0[1])):
      print('HITId: ', results_0[2][i])
      print('post: ', dataset['post'][i])
      print('Enc Dec 1', results_0[1][i])
      print('Enc Dec 2', results_1[1][i])
      #if results_0[1][i] != results_1[1][i]:
      #  num_diff += 1
      #  hitid_diff.append(results_0[2][i])
    #print("Num Different: ", num_diff)
    #print("Corresponding HITIds: ", hitid_diff)
    
    ### Print Model Output ###
    #print("generating sample outputs ...")
    #print_example_outputs(tokenized, seq2seq_tok, model, use_cuda=args.use_cuda)
