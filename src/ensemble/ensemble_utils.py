import sys
sys.path.append('../../shared/')

import pickle
import pandas as pd
import numpy as np

from utils import *
from transformers import AutoModel
from collections import defaultdict

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

ENSEMBLE_TRAIN_DATA_FILE = '../../data/SBIC.v2.ensemble_trn.csv'
ENSEMBLE_DEV_DATA_FILE = '../../data/SBIC.v2.ensemble_dev.csv'

MAX_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128
SEP_TOKEN = '</s>'
EMPTY_TOKEN = '</e>'

def add_view_token_idx_col(tokenized, token_ids, train=True):
    def add_view_token_idx_row(row):
      row['view_token_idx'] = [i for i,input_id in enumerate(row['input_ids']) if input_id in token_ids]
      return row
    
    if train:
      for dataset in tokenized:
        tokenized[dataset] = tokenized[dataset].map(add_view_token_idx_row)
    else:
      tokenized = tokenized.map(add_view_token_idx_row)
    
    return tokenized

def sort_row(x):
    x = x.split(SEP_TOKEN)
    x = list(map(lambda y: y.strip('!?. '), x))
    x.sort()
    return x

def get_sent_counts(x):
    sent_prev = x[0]
    sent_counts = [[1, sent_prev]]
  
    for sent_curr in x[1:]:
      if sent_curr == sent_prev:
        sent_counts[-1][0] += 1
      else:
        sent_counts.append([1, sent_curr])
      sent_prev = sent_curr
  
    sent_counts.sort(reverse=True)
    return sent_counts

def sent_counts_to_str_count_group(sent_counts):
    counts = ''
    sentences = ''
    
    for i,sent_count in enumerate(sent_counts):
      count_str = str(sent_count[0])
      counts += count_str
      
      if i == len(sent_counts) - 1:
        sentences += sent_count[1]
      else:
        sentences += sent_count[1] + SEP_TOKEN
    
    x = counts + SEP_TOKEN + sentences
    return x

def sent_counts_to_str_ordered(sent_counts):
    x = ''
    
    for i,sent_count in enumerate(sent_counts):
      if i == len(sent_counts) - 1:
        x += sent_count[1]
      else:
        x += sent_count[1] + SEP_TOKEN
    
    return x

def sent_counts_to_str_dist(sent_counts, total_sents):
    x = ''
    total_rat = 0.0

    for i,sent_count in enumerate(sent_counts):
      total_rat += sent_count[0] / float(total_sents)
      count_str = "{:.2f}".format(sent_count[0] / float(total_sents))
      
      if i == len(sent_counts) - 1:
        x += count_str + SEP_TOKEN + sent_count[1]
      else:
        x += count_str + SEP_TOKEN + sent_count[1] + SEP_TOKEN
    
    return x

def sent_counts_to_str_count(sent_counts):
    x = ''
    
    for i,sent_count in enumerate(sent_counts):
      count_str = str(sent_count[0])
      
      if i == len(sent_counts) - 1:
        x += count_str + SEP_TOKEN + sent_count[1]
      else:
        x += count_str + SEP_TOKEN + sent_count[1] + SEP_TOKEN
    
    return x

# Gets string with sentence counts for the sorted BART Ensemble Model variants
def get_sent_counts_str_ordered(x):
    x = sort_row(x)
    sent_counts = get_sent_counts(x)
    x = sent_counts_to_str_ordered(sent_counts)
    return x

# Gets string with sentence counts for the sorted BART Ensemble Model variants
def get_sent_counts_str_count_group(x):
    x = sort_row(x)
    sent_counts = get_sent_counts(x)
    x = sent_counts_to_str_count_group(sent_counts)
    return x

# Gets string with sentence counts for the sorted BART Ensemble Model variants
def get_sent_counts_str_dist(x):
    x = sort_row(x)
    total_sents = len(x)

    sent_counts = get_sent_counts(x)
    x = sent_counts_to_str_dist(sent_counts, total_sents)
    return x

# Gets string with sentence counts for the sorted BART Ensemble Model variants
def get_sent_counts_str_count(x):
    x = sort_row(x)
    total_sents = len(x)

    sent_counts = get_sent_counts(x)
    x = sent_counts_to_str_count(sent_counts)
    return x

# Gets string with sentence counts for the unsorted BART Ensemble Model
def get_sent_counts_str_unordered_count(x):
    x = x.split(SEP_TOKEN)
    x = list(map(lambda y: y.strip('!?. '), x))
    counts = defaultdict(int)
    
    for sent in x:
      counts[sent] += 1
    
    count_str = ''
    sent_str = ''
    for i,sent in enumerate(x):
      count_str += str(counts[sent])
      if i == len(x) - 1:
        sent_str += sent
      else:
        sent_str += sent + SEP_TOKEN
    
    x = count_str + SEP_TOKEN + sent_str
    return x

def get_view_token(view_idx, block_idx, view_token_set=None):
    view_token = '</vw' + str(view_idx) + str(block_idx) + '>'
    if view_token_set is not None:
      view_token_set.add(view_token)
    return view_token

# Here, the model index is equivalent to the view index
def concat_model_views_1(df, model_idx, num_views, view_tokens, offset=0):
    for i in range(offset, num_views + offset):
      vw_block_0 = get_view_token(i, 0, view_tokens)
      view_col_name = 'view_' + str(i)
      
      if i - offset == model_idx:
        vw_block_1 = get_view_token(i, 1, view_tokens)
        df[view_col_name] = vw_block_0 + vw_block_1 + df['prediction']
      else:
        df[view_col_name] = vw_block_0 + df['prediction']

# Here, the model index is equivalent to the view index
def concat_model_views_2(df, model_idx, num_views, view_tokens, offset=0):
    for i in range(offset, num_views + offset):
      view_col_name = 'view_' + str(i)
      if i - offset == model_idx - 1:
        vw_block_2 = get_view_token(i, 2, view_tokens)
        df[view_col_name] = df[view_col_name] + vw_block_2 + df['prediction_0']
      elif i - offset == model_idx:
        vw_block_1 = get_view_token(i, 1, view_tokens)

        if i - offset == 5:
          vw_block_2 = get_view_token(i, 2, view_tokens)
          df[view_col_name] = df[view_col_name] + vw_block_1 + df['prediction_0'] + vw_block_2
        else:
          df[view_col_name] = df[view_col_name] + vw_block_1 + df['prediction_0']
      else:
        df[view_col_name] = df[view_col_name] + SEP_TOKEN + df['prediction_0']

def concat_knowledge_views_1(df, model_idx, num_views, view_tokens, offset=0):
    concat_model_views_1(df, model_idx, num_views, view_tokens, offset=offset)

# Here, the model index is not equivalent to the view index
def concat_knowledge_views_2(df, model_idx, num_views, view_tokens, offset=0):
    view_idx = model_idx // 2
    for i in range(offset, num_views + offset):
      view_col_name = 'view_' + str(i)
      if model_idx % 2 == 1:
        if model_idx == 5 and i == num_views + offset - 1:
          vw_block_2 = get_view_token(i, 2, view_tokens)
          df[view_col_name] = df[view_col_name] + SEP_TOKEN + df['prediction_0'] + vw_block_2
        else:
          df[view_col_name] = df[view_col_name] + SEP_TOKEN + df['prediction_0']
      elif i - offset == view_idx - 1:
        vw_block_2 = get_view_token(i, 2, view_tokens)
        df[view_col_name] = df[view_col_name] + vw_block_2 + df['prediction_0']
      elif i - offset == view_idx:
        vw_block_1 = get_view_token(i, 1, view_tokens)
        df[view_col_name] = df[view_col_name] + vw_block_1 + df['prediction_0']
      else:
        df[view_col_name] = df[view_col_name] + SEP_TOKEN + df['prediction_0']

# Still need to add logic for use empty token
def read_pickle_files(ensemble_files, model_type='vanilla_ensemble', num_views=6):
    df = pickle.load(open(ensemble_files[0], 'rb'))[['HITId','prediction']]
    model_idx = 0
    view_tokens = set()

    df['prediction'] = df['prediction'].apply(lambda col: ' '.join(col.split()[:MAX_OUTPUT_LENGTH]))
    if model_type == 'vanilla_ensemble_view':
      if num_views == 3:
        concat_knowledge_views_1(df, model_idx, num_views, view_tokens)
      elif num_views == 6:
        concat_model_views_1(df, model_idx, num_views, view_tokens)
      elif num_views == 9:
        concat_model_views_1(df, model_idx, 6, view_tokens)
        concat_knowledge_views_1(df, model_idx, 3, view_tokens, offset=6)

    for ensemble_file in ensemble_files[1:]:
      df_i = pickle.load(open(ensemble_file, 'rb'))[['HITId', 'prediction']]
      model_idx += 1

      df_i.rename(columns={'prediction':'prediction_0'}, copy=False, inplace=True)
      df = df.merge(df_i, on='HITId', validate='one_to_one')
      df['prediction_0'] = df['prediction_0'].apply(lambda col: ' '.join(col.split()[:MAX_OUTPUT_LENGTH]))
      
      if model_type == 'vanilla_ensemble_view':
        if num_views == 3:
          concat_knowledge_views_2(df, model_idx, num_views, view_tokens)
        elif num_views == 6:
          concat_model_views_2(df, model_idx, num_views, view_tokens)
        elif num_views == 9:
          concat_model_views_2(df, model_idx, 6, view_tokens)
          concat_knowledge_views_2(df, model_idx, 3, view_tokens, offset=6)
          
      else:
          df['prediction'] = df['prediction'] + SEP_TOKEN + df['prediction_0']

      df.drop(['prediction_0'], axis=1, inplace=True)
    
    view_tokens = list(view_tokens)
    view_tokens.sort()
    if model_type == 'vanilla_ensemble' or model_type == 'vanilla_ensemble_view':
      pass
    elif model_type == 'ordered':
      df['prediction'] = df['prediction'].apply(get_sent_counts_str_ordered)
    elif model_type == 'dist':
      df['prediction'] = df['prediction'].apply(get_sent_counts_str_dist)
    elif model_type == 'count':
      df['prediction'] = df['prediction'].apply(get_sent_counts_str_count)
    elif model_type == 'count_group':
      df['prediction'] = df['prediction'].apply(get_sent_counts_str_count_group)
    elif model_type == 'unordered_count':
      df['prediction'] = df['prediction'].apply(get_sent_counts_str_unordered_count)

    return df, view_tokens

