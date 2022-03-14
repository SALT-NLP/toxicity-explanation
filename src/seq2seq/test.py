import pandas as pd
import numpy as np
import os
import math
import argparse
from seq2seq_utils import *
from seq2seq import BartForConditionalGenerationJoinModel
from torch import nn, torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from train import *

# Overwrite standard warning output with new warning output
import warnings
warnings.formatwarning = custom_warning

# Useful constants
CLASSIFIER_TOK_NAME = 'bert-base-uncased'
SEQ2SEQ_TOK_NAME = 'facebook/bart-base'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
        help='The path for the classifier. You can pass multiple, but they must be passed in the same order used when training',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    parser.add_argument('--sep', type=str, default=',', help='Pass in a separator for the data file.')
    parser.add_argument('--generate_scores', action='store_true', help='If True, will generate scores')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--hitid_file', help='Path to HITID file for generation.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.dev.csv', help='Data File to load.')

    return parser.parse_args()

def check_args(args):
    use_hitid = args.hitid_file is not None
    model_path = args.model
    model_name = get_file_name(model_path)
    
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')
    
    data_source = get_file_name(args.data_file)
    pickle_file = 'pred/' + model_name + '_' + data_source
    pickle_file = pickle_file + '.pickle' if args.hitid_file is None else pickle_file + '_hitid.pickle'
    results_file = 'results/' + model_name + '_' + data_source + '.csv'

    join = '_join_' in model_name
    
    if join and args.classifiers is None:
      raise ValueError('You have selected a join model, but have not provided any classifiers as args.')
    
    return model_path, pickle_file, results_file, join

if __name__ == '__main__':
    # Parse Args
    args = parse_args()
    model_path, pickle_file, results_file, join = check_args(args)
    print(results_file)

    if not os.path.exists(pickle_file):
      # Load hitids if the file was passed in
      hitid_set = None
      if args.hitid_file is not None:
        hitid_set = get_hitids(args.hitid_file)
      
      # Tokenize Data
      print("cleaning csv ...")
      dataset = read_and_clean_csv(args.data_file, train=False, post_ids=hitid_set, sep=args.sep)
      
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
      seq2seq_tok, dataset = tokenize_textgen_df(
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
      
      # Initiailize Minibatch Iterator
      input_cols = ['input_ids', 'classifier_attention', 'attention_mask'] if join else ['input_ids', 'attention_mask']
      batch_iter = MinibatchIterator(dataset, seq2seq_tok, batch_size=args.batch_size, torch_cols=input_cols, use_cuda=args.use_cuda)
      enc_forward = model.encoder_enrichment_forward if join else model.get_encoder().forward

      # Run Tests
      print('running model tests ...')
      generate_stereotypes(
          batch_iter,
          seq2seq_tok,
          model,
          enc_forward,
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

