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
CLASS_COL = [
    'HITId', 'post', 'sexYN', 'offensiveYN', 'intentYN', 'whoTarget', \
    'speakerMinorityYN'
]
LANG_GEN_COL = ['HITId', 'targetMinority', 'targetStereotype']

GPT_DICT = {
    'TRAINED MODEL': 'model/gpt_baseline',
    'BASE MODEL': 'openai-gpt',
}

GPT2_DICT = {
    'TRAINED MODEL': 'model/gpt2_baseline',
    'BASE MODEL': 'gpt2',
}

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model_type', choices=['gpt','gpt2'], default='gpt', help='Pass either \'gpt\' or \'gpt2\'.')
  parser.add_argument('--data_file', type=str, default='../data/SBIC.v2.dev.csv', help='Data File to load.')
  parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
  parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
  parser.add_argument('--predict', action='store_true', help='Whether or not to run predictions. If False, will look for a prediction file.')
  parser.add_argument('--hitid_file', help='Path to HITID file for generation.')
  parser.add_argument('--num_beams', type=int, default=1, help='Set the number of beams for Beam Search.')
  parser.add_argument('--max_length', type=int, default=128, help='Maximum Length Sequence to generate.')
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

  pred_file_name = 'pred/' + args.model_type + '_' + data_source
  pred_file_name = pred_file_name + '_hitid' if use_hitid else pred_file_name
  results_file = 'results/' + args.model_type + '_' + data_source + '.csv'

  active_dict = get_active_dict(args)
  active_dict['TO ACTUAL'] = pred_file_name + '_actual.pickle'
  active_dict['TO PRED'] = pred_file_name + '_pred.pickle'
  active_dict['RESULTS'] = results_file
    
  if not(os.path.isfile(args.data_file)):
    raise ValueError('Must pass in an existing data file for testing.')

  if args.predict:
    if os.path.isfile(active_dict['TO ACTUAL']):
      warnings.warn(active_dict['TO ACTUAL'] + ' exists and will be overwritten', RuntimeWarning)
    if os.path.isfile(active_dict['TO PRED']):
      warnings.warn(active_dict['TO PRED'] + ' exists and will be overwritten', RuntimeWarning)
  else:
    if not os.path.isfile(active_dict['TO ACTUAL']):
      raise ValueError(active_dict['TO ACTUAL'] + ' does not exist. Run again with predict flag set to True')
    if not os.path.isfile(active_dict['TO PRED']):
      raise ValueError(active_dict['TO PRED'] + ' does not exist. Run again with predict flag set to True')
    if use_hitid:
      warnings.warn(hitid_file + ' may not be used to filter data, since predict flag was not passed.')

  return active_dict

# Constants defined in gpt_utils
if __name__ == "__main__":
  args = parse_args()
  active_dict = check_args(args)

  print('cleaning data ...')
  df = pd.read_csv(args.data_file)
  df = clean_post(df)

  if args.predict:
    print('initializing model and tokenizer ...')
    tokenizer = setup_tokenizer(active_dict['BASE MODEL'])
    model = AutoModelForCausalLM.from_pretrained(active_dict['TRAINED MODEL'], \
                                                 pad_token_id=tokenizer.eos_token_id)
    model.eval()
    
    hitid_set = None
    if args.hitid_file is not None:
      hitid_set = get_hitids(args.hitid_file)

    print('predicting on data ...')
    actual = get_samples_from_actual(df, PRED_COL, post_ids=hitid_set)
    print(len(actual))
    predict_samples(model, tokenizer, actual, PRED_COL, active_dict, args.max_length, args.num_beams)
  
  print('computing scores ...')
  actual = pickle.load(open(active_dict['TO ACTUAL'], 'rb'))
  pred = pickle.load(open(active_dict['TO PRED'], 'rb'))  

  if args.save_results_to_csv:
    save_results_to_csv(pred, actual, num_results=args.num_results, save_file=active_dict['RESULTS'])
  else:
    get_and_print_f1_scores(actual, pred)
    get_and_print_lang_gen_scores(df, actual, pred)
