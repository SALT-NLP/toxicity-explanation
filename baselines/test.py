import sys
sys.path.append('../shared/')

import torch
import argparse
import os
import pandas as pd
import numpy as np

from gpt_utils import *
from utils import *
from datasets import Dataset
from transformers import AutoModelForCausalLM

MAX_LENGTH = 128
FROM_FILE = '../data/SBIC.v2.dev.csv'
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
    'TO ACTUAL': 'pred/gpt_5epoch_dev_actual.csv',
    'TO PRED': 'pred/gpt_5epoch_dev_pred.csv',
    'TRAINED MODEL': 'model/gpt_baseline',
    'BASE MODEL': 'openai-gpt',
}

GPT2_DICT = {
    'TO ACTUAL': 'pred/gpt2_5epoch_dev_actual.csv',
    'TO PRED': 'pred/gpt2_5epoch_dev_pred.csv',
    'TRAINED MODEL': 'model/gpt2_baseline',
    'BASE MODEL': 'gpt2',
}

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

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', choices=['gpt','gpt2'], default='gpt', help='Pass either \'gpt\' or \'gpt2\'')
  return parser.parse_args()

def set_args(args):
  if args.model_type == 'gpt':
    active_dict = GPT_DICT
  else:
    active_dict = GPT2_DICT

  return active_dict

# Constants defined in gpt_utils
def get_and_print_f1_scores(actual, pred):
  print("Category: (Precision, Recall, F1)")
  print('Offensive: ', f1_score(actual, pred, 'offensiveYN', OFFY, OFFN))
  print('Intent: ', f1_score(actual, pred, 'intentYN', INTY, INTN))
  print('Lewd: ', f1_score(actual, pred, 'sexYN', LEWDY, LEWDN))
  print('Group Targeted: ', f1_score(actual, pred, 'whoTarget', GRPY, GRPN))
  print('In Group: ', f1_score(actual, pred, 'speakerMinorityYN', INGY, INGN))

def get_and_print_lang_gen_scores(df, actual, pred):
  sub_df = df[LANG_GEN_COL]
  sub_df = aggregate_and_format(sub_df)
  actual = actual[CLASS_COL].join(sub_df, on='HITId').reindex(columns=PRED_COL)
  
  references_tm, hypotheses_tm = get_references_and_hypotheses('targetMinority', actual, pred)
  bleu_score_tm_max, bleu_score_tm_avg = get_bleu_score(references_tm, hypotheses_tm)
  rouge_scores_tm_max, rouge_scores_tm_avg = get_rouge_scores(references_tm, hypotheses_tm)
  
  references_ts, hypotheses_ts = get_references_and_hypotheses('targetStereotype', actual, pred)
  bleu_score_ts_max, bleu_score_ts_avg = get_bleu_score(references_ts, hypotheses_ts)
  rouge_scores_ts_max, rouge_scores_ts_avg = get_rouge_scores(references_ts, hypotheses_ts)
  
  print("Target Minority Scores: ")
  print("Bleu Score (Avg): ", bleu_score_tm_avg)
  print("Bleu Score (Max): ", bleu_score_tm_max)
  print("Rouge Score (Avg) (Precision, Recall, F1): ", rouge_scores_tm_avg)
  print("Rouge Score (Max) (Precision, Recall, F1): ", rouge_scores_tm_max)
  
  print("Implied Stereotype Scores: ")
  print("Bleu Score (Avg): ", bleu_score_ts_avg)
  print("Bleu Score (Max): ", bleu_score_ts_max)
  print("Rouge Score (Avg) (Precision, Recall, F1): ", rouge_scores_ts_avg)
  print("Rouge Score (Max) (Precision, Recall, F1): ", rouge_scores_ts_max)

if __name__ == "__main__":
  args = parse_args()
  active_dict = set_args(args)
  
  print('cleaning data ...')
  df = pd.read_csv(FROM_FILE)
  clean_post(df)

  if not os.path.exists(active_dict['TO ACTUAL']) or not os.path.exists(active_dict['TO PRED']):
    print('initializing model and tokenizer ...')
    tokenizer = setup_tokenizer(active_dict['BASE MODEL'])
    model = AutoModelForCausalLM.from_pretrained(active_dict['TRAINED MODEL'], \
                                                 pad_token_id=tokenizer.eos_token_id)
    model.eval()
    
    print('predicting on data ...')
    actual = get_samples_from_actual(df, PRED_COL, post_ids=HITID_SET)
    predict_samples(model, tokenizer, actual, PRED_COL, active_dict, MAX_LENGTH)
  
  print('computing scores ...')
  actual = pd.read_csv(active_dict['TO ACTUAL'])
  pred = pd.read_csv(active_dict['TO PRED'])
  
  get_and_print_f1_scores(actual, pred)
  get_and_print_lang_gen_scores(df, actual, pred)

