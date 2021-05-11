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
CLASSIFIER_MODEL_NAME = 'bert-base-uncased'
CLASSIFIERS = ['./classification/model/offensiveYN/checkpoint-1798/']
#CLASSIFIERS = ['./classification/model/offensiveYN/checkpoint-898/',
#               './classification/model/intentYN/checkpoint-898/',
#               './classification/model/sexYN/checkpoint-898/',
#               './classification/model/whoTarget/checkpoint-1280/']

SEQ2SEQ_TOK_NAME = 'facebook/bart-base'

#SEQ2SEQ_BART_BASE = './model/bart_base_checkpoint-3epoch/'
#SEQ2SEQ_PICKLE_BASE = './data/bart_base_checkpoint-3epoch.pickle'

#SEQ2SEQ_BART_JOIN = './model/bart_join_pretrain_zeros_group_checkpoint-21560/'
#SEQ2SEQ_PICKLE_JOIN = './data/bart_join_pretrain_zeros_group_checkpoint-21560.pickle'

DATA_DIR = '../data/'
FROM_DATA = DATA_DIR + 'SBIC.v2.dev.csv'
TO_DATA = 'data/clean_dev_df.csv'

def generate_stereotypes(tokenized, seq2seq_tok, model, pickle_file, use_cuda=True):
    batch_size = 20
    results = [[],[]]
    num_batches = math.ceil(tokenized.num_rows // batch_size)

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
      '--model',
      action='store',
      type=str,
      required=True,
      help='Send the checkpoint folder name in model/'
    )
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse Args
    args = parse_args()
    model_name = 'model/' + args.model + '/'
    pickle_file = 'data/' + args.model + '.pickle'
    join = '_join_' in args.model
    
    if not os.path.exists(pickle_file):
      # Tokenize Data
      print('preparing and tokenizing data ...')
      seq2seq_tok, classifier_tok, tokenized = tokenize_df(
          FROM_DATA,
          TO_DATA,
          SEQ2SEQ_TOK_NAME,
          CLASSIFIER_MODEL_NAME,
          train=False,
      )
      
      # Initialize Model
      print('initializing model ...')
      model = init_model(
        model_name,
        join=join,
        classifiers=CLASSIFIERS,
        train=False,
        use_cuda=args.use_cuda
      )
      
      # Run Tests
      print('running model tests ...')
      generate_stereotypes(tokenized, seq2seq_tok, model, pickle_file, use_cuda=args.use_cuda)
    
    print("generating base model scores ...")
    generate_scores(pickle_file)

    ### Print Model Output ###
    #print("generating sample outputs ...")
    #print_example_outputs(tokenized, seq2seq_tok, model, use_cuda=args.use_cuda)
