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

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'
CONCEPTNET_EMBEDDING_FILE = 'data/numberbatch-en-19.08.txt'

EDGE_DATA_FILE = 'data/conceptnet-assertions-5.7.0.csv'
EDGE_DICT_FILE = 'data/edge_dictionary.pickle'
EMB_DICT_FILE = 'data/embedding_dictionary.pickle'

BART_HIDDEN_SIZE = 768
EMBEDDING_SIZE = 300
MAX_LENGTH = 1024

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='The path for the checkpoint folder',
    )
    
    parser.add_argument('--k', type=int, default=5, help='Pass in a value for k.')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--sep', type=str, default=',', help='Separator for data file.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    parser.add_argument('--generate_scores', action='store_true', help='If True, will generate scores')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--model_type', type=str, choices=['input', 'attn'], required=True, help='Pass in a model type.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.dev.csv', help='Data File to load.')
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
    
    return model_path, pickle_file, results_file

if __name__ == '__main__':
    args = parse_args()
    model_path, pickle_file, results_file = check_args(args)
    
    if not os.path.exists(pickle_file):
      print('loading and tokenizing data ...')
      df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
      df = clean_post(df)
      df = clean_target(df, train=False)
      
      df_post = df[['HITId', 'post']]
      df_post = df_post.drop_duplicates().reset_index(drop=True)
      
      print('processing edge file ...')
      if not os.path.exists(EDGE_DICT_FILE):
        edge_dict = process_edge_file(EDGE_DATA_FILE)
        pickle.dump(edge_dict, open(EDGE_DICT_FILE, 'wb'))
      else:
        edge_dict = pickle.load(open(EDGE_DICT_FILE, 'rb'))
    
      print('processing numberbatch embeddings ...')
      emb_dict = None
      if args.model_type == 'attn':
        if not os.path.exists(EMB_DICT_FILE):
          emb_dict = process_embedding_file(CONCEPTNET_EMBEDDING_FILE)
          pickle.dump(emb_dict, open(EMB_DICT_FILE, 'wb'))
        else:
          emb_dict = pickle.load(open(EMB_DICT_FILE, 'rb'))
      
      print('finding top k tuples ...')
      df_post = concat_top_k_tuples(df_post, edge_dict, emb_dict=emb_dict, emb_size=EMBEDDING_SIZE*2, k=args.k)
      df = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')
      dataset = Dataset.from_pandas(df)
      seq2seq_tok, tokenized = tokenize_bart_df(dataset, SEQ2SEQ_MODEL_NAME, padding=False, train=False, max_length=MAX_LENGTH)
      
      print('initializing model ...')
      if args.model_type == 'input':
        model = BartForConditionalGeneration.from_pretrained(model_path)
        forward_method = model.get_encoder().forward
        torch_cols = ['input_ids', 'attention_mask']
      elif args.model_type == 'attn':
        model = BartForConditionalGenerationKnowledgeModel.from_pretrained(model_path)
        forward_method = model.encoder_knowledge_forward
        torch_cols = ['input_ids', 'knowledge_embeds', 'attention_mask']
      model.eval()
      if args.use_cuda and torch.cuda.is_available():
        model.cuda()
      
      print('running model tests ...')
      batch_iter = MinibatchIterator(tokenized, seq2seq_tok, batch_size=args.batch_size, torch_cols=torch_cols, use_cuda=args.use_cuda)
      generate_stereotypes(
          batch_iter,
          seq2seq_tok,
          model,
          forward_method,
          results_cols=['HITId','post','target'],
          pickle_file=pickle_file,
      )
    
    if args.generate_scores or args.save_results_to_csv:
      print("generating base model scores ...")
      generate_scores(
        pickle_file,
        save_results_to_csv=args.save_results_to_csv,
        num_results=args.num_results,
        save_file=results_file
      )

