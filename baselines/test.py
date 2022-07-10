import sys
sys.path.append('../shared/')

import torch
import argparse
import os
import pickle
import pandas as pd
import numpy as np

from gpt_utils import *
from utils import *
from datasets import Dataset
from transformers import AutoModelForCausalLM

# Overwrite standard warning output with new warning output
import warnings
warnings.formatwarning = custom_warning

PRED_COL = [
    'HITId', 'post', 'sexYN', 'offensiveYN', 'intentYN', 'whoTarget', \
    'targetMinority','targetStereotype', 'speakerMinorityYN'
]

PRED_COL_IMPL = [
    'HITId', 'post', 'targetMinority', 'targetStereotype'
]

GPT_DICT = {
    'BASE MODEL': 'openai-gpt',
}

GPT2_DICT = {
    'BASE MODEL': 'gpt2',
}

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model_type', choices=['gpt','gpt2'], default='gpt', help='Pass either \'gpt\' or \'gpt2\'.')
  parser.add_argument('--model_file', type=str, default=None, help='Model file to load.')
  parser.add_argument('--data_file', type=str, default='../data/SBIC.v2.dev.csv', help='Data File to load.')
  parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
  parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
  parser.add_argument('--hitid_file', help='Path to HITID file for generation. WARNING THIS HASN''T BEEN TESTED AND MAY NOT WORK!!')
  parser.add_argument('--num_beams', type=int, default=1, help='Set the number of beams for Beam Search.')
  parser.add_argument('--max_length', type=int, default=128, help='Maximum Length Sequence to generate.')
  parser.add_argument('--impl_hate_data', action='store_true', help='Are you using the implicit hate dataset?')
  parser.add_argument('--sep', default=',', help='Separator for file read.')
  return parser.parse_args()
  
def get_active_dict(args):
  if args.model_type == 'gpt':
    active_dict = GPT_DICT
  else:
    active_dict = GPT2_DICT

  return active_dict

def save_results_to_csv(pred, actual, generation_seed=756, num_results=200, save_file='results.csv'):
  indices = list(range(len(pred['post'])))
  random.seed(generation_seed)
  random.shuffle(indices)
  
  col_names = ['HITId', 'post', 'target','prediction']
  results_csv = []
  for idx in indices[:num_results]:
    results_csv.append([
      pred['HITId'][idx],
      pred['post'][idx].replace('\n', ' '),
      ', '.join(actual['targetStereotype'][idx]),
      pred['targetStereotype'][idx],
    ])
  
  with open(save_file, 'w') as f:
    csv_writer = csv.writer(f, delimiter='|')
    csv_writer.writerow(col_names)
    csv_writer.writerows(results_csv)

def check_args(args):
  use_hitid = args.hitid_file is not None
  data_source = get_file_name(args.data_file)
  model_name = get_file_name(args.model_file)

  pred_file_name = 'pred/' + model_name
  pred_file_name = pred_file_name + '_hitid' if use_hitid else pred_file_name
  results_file = 'results/' + model_name + '.csv'

  active_dict = get_active_dict(args)
  active_dict['TRAINED MODEL'] = args.model_file
  active_dict['TO ACTUAL'] = pred_file_name + '_actual.pickle'
  active_dict['TO PRED'] = pred_file_name + '_pred.pickle'
  active_dict['RESULTS'] = results_file
    
  if not(os.path.isfile(args.data_file)):
    raise ValueError('Must pass in an existing data file for testing.')

  return active_dict

# Constants defined in gpt_utils
if __name__ == "__main__":
  args = parse_args()
  active_dict = check_args(args)
  print('Args: ', args)

  print('cleaning data ...')
  df = pd.read_csv(args.data_file, sep=args.sep, engine='python')[:50]
  df = clean_post(df)
  df.targetMinority = df.targetMinority.replace(np.nan, '', regex=True)
  df.targetStereotype = df.targetStereotype.replace(np.nan, '', regex=True)

  if not os.path.isfile(active_dict['TO ACTUAL']) or not os.path.isfile(active_dict['TO PRED']):
    print('initializing model and tokenizer ...')
    tokenizer = setup_tokenizer(active_dict['BASE MODEL'])
    model = AutoModelForCausalLM.from_pretrained(active_dict['TRAINED MODEL'], \
                                                 pad_token_id=tokenizer.eos_token_id)
    model.eval()
    
    hitid_set = None
    if args.hitid_file is not None:
      hitid_set = get_hitids(args.hitid_file)

    print('predicting on data ...')
    pred_col = PRED_COL_IMPL if args.impl_hate_data else PRED_COL
    actual = get_samples_from_actual(df, pred_col, post_ids=hitid_set, impl=args.impl_hate_data)
    
    if args.impl_hate_data:
      predict_samples_impl(model, tokenizer, actual, PRED_COL, active_dict, args.max_length, args.num_beams)
    else:
      predict_samples(model, tokenizer, actual, PRED_COL, active_dict, args.max_length, args.num_beams)
  
  print('computing scores ...')
  actual = pickle.load(open(active_dict['TO ACTUAL'], 'rb'))
  pred = pickle.load(open(active_dict['TO PRED'], 'rb'))  
  
  if args.save_results_to_csv:
    save_results_to_csv(pred, actual, num_results=args.num_results, save_file=active_dict['RESULTS'])
  else:
    if not args.impl_hate_data:
      get_and_print_f1_scores(actual, pred)
    get_and_print_lang_gen_scores(df, actual, pred)
