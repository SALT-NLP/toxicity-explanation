import sys
sys.path.append('../../shared/')

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

def generate_stereotypes(tokenized, seq2seq_tok, model, pickle_file, batch_size=16, use_cuda=True):
    results = [[],[]]
    num_batches = math.ceil(tokenized.num_rows / batch_size)

    for batch in tqdm(range(num_batches)):
      i = batch * batch_size
      j = min(tokenized.num_rows, i + batch_size)
      
      _, output_strs = generate_batch(tokenized, seq2seq_tok, model, i, j, use_cuda=use_cuda)
      results[0].extend(tokenized['target'][i:j])
      results[1].extend(output_strs)
    pickle.dump(results, open(pickle_file, 'wb'))

def generate_scores(pickle_file):
    results = pickle.load(open(pickle_file, 'rb'))
    references = results[0]
    hypotheses = results[1]

    bleu_score_max, bleu_score_avg = get_bleu_score(references, hypotheses)
    rouge_scores_max, rouge_scores_avg = get_rouge_scores(references, hypotheses)

    metric = load_metric('bertscore')
    bert_scores = metric.compute(predictions=hypotheses, references=references, lang='en')
    bert_score = get_bert_score(bert_scores, hypotheses, references)

    print("Bleu Score (Avg): ", bleu_score_avg)
    print("Bleu Score (Max): ", bleu_score_max)
    print("Rouge Score (Avg) (Precision, Recall, F1): ", rouge_scores_avg)
    print("Rouge Score (Max) (Precision, Recall, F1): ", rouge_scores_max)
    print('BERT Score (Max) (Precision, Recall, F1): ', bert_score)

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
        '--model',
        required=True,
        help='The path for the checkpoint folder',
    )
    parser.add_argument(
        '-c',
        '--classifiers',
        nargs='+',
        required=False,
        help='The path for the classifier. You can pass multiple, but they must be passed in the same order as in training',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse Args
    args = parse_args()
    model_path = args.model
    model_name = os.path.basename(os.path.normpath(model_path))
    
    pickle_file = 'data/' + model_name + '.pickle'
    join = '_join_' in model_name
    
    if not os.path.exists(pickle_file):
      if join and args.classifiers is None:
        raise ValueError('You have selected a join model, but have not provided any classifiers as args.')
    
      # Tokenize Data
      print("cleaning csv ...")
      dataset = read_and_clean_csv(FROM_DATA_FILE, TO_DATA_FILE, train=False)

      num_classifiers = 0
      num_classification_heads = 0
      
      if join:
        print("tokenizing and classifying data ...")
        attentions = get_classifier_attention(dataset, CLASSIFIER_TOK_NAME, args.classifiers, args.use_cuda)
        
        print("mapping classifier attention to dataset ...")
        dataset = map_column_to_dataset(dataset, attentions, 'classifier_attention')
        
        num_classifiers = attentions.shape[1]
        num_classification_heads = attentions.shape[2]
      
      print('tokenizing data for bart ...')
      seq2seq_tok, dataset = tokenize_bart_df(
          dataset,
          SEQ2SEQ_TOK_NAME,
          train=False
      )
      
      # Initialize Model
      print('initializing model ...')
      model = init_model(
          model_path,
          join=join,
          num_classifiers=num_classifiers,
          num_classification_heads=num_classification_heads,
          train=False,
          use_cuda=args.use_cuda
      )
      
      # Run Tests
      print('running model tests ...')
      generate_stereotypes(
        dataset,
        seq2seq_tok,
        model,
        pickle_file,
        batch_size=args.batch_size,
        use_cuda=args.use_cuda
      )
    
    print("generating base model scores ...")
    generate_scores(pickle_file)

    ### Print Model Output ###
    #print("generating sample outputs ...")
    #print_example_outputs(tokenized, seq2seq_tok, model, use_cuda=args.use_cuda)
    
