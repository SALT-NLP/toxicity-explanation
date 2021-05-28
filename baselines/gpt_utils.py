"""
This file contains data cleaning functions for the SBIC corpus

See the SBIC Corpus here:
  https://homes.cs.washington.edu/~msap/social-bias-frames/DATASTATEMENT.html
"""
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from tqdm import tqdm

# String Separator
SEP = '[SEP]'
BOS = '[STR]'
EOS = '[END]'
PAD = '[PAD]'

# Categorical Special Tokens
LEWDY = '[lewdY]'
LEWDN = '[lewdN]'
OFFY = '[offY]'
OFFN = '[offN]'
INTY = '[intY]'
INTN = '[intN]'
GRPY = '[grpY]'
GRPN = '[grpN]'
INGY = '[ingY]'
INGN = '[ingN]'

######################### Training Utils #########################
def clean_post(df):
  #df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
  #df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
  df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
  df.post = df.post.str.strip()
  return df

def categorize_var(df):
  df.sexYN = np.where(df.sexYN >= 0.5, LEWDY, LEWDN)
  df.offensiveYN = np.where(df.offensiveYN >= 0.5, OFFY, OFFN)
  df.intentYN = np.where(df.intentYN >= 0.5, INTY, INTN)
  df.whoTarget = np.where(df.whoTarget >= 0.5, GRPY, GRPN)
  df.speakerMinorityYN = np.where(df.speakerMinorityYN >= 0.5, INGY, INGN)
  return df

def create_text_column(df):
  df = categorize_var(df)
  df['text'] = BOS + df.post + SEP + df.sexYN + ' ' + df.offensiveYN + ' ' + \
                  df.intentYN + ' ' + df.whoTarget + SEP + df.targetMinority + \
                  SEP + df.targetStereotype + SEP + df.speakerMinorityYN + EOS

def clean_df(from_file, to_file):
  df = pd.read_csv(from_file)
  clean_post(df)
  df.targetMinority = df.targetMinority.replace(np.nan, '', regex=True)
  df.targetStereotype = df.targetStereotype.replace(np.nan, '', regex=True)
  
  create_text_column(df)
  df[['text']].sample(frac=1).to_csv(to_file, index=False)
  return Dataset.from_pandas(df[['text']])

def setup_tokenizer(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
  categorical_tokens = [LEWDY,LEWDN,OFFY,OFFN,INTY,INTN,GRPY,GRPN,INGY,INGN]
  special_tokens = {'sep_token': SEP, \
                    'bos_token': BOS, \
                    'eos_token': EOS, \
                    'pad_token': PAD, \
                    'additional_special_tokens': categorical_tokens}
  tokenizer.add_special_tokens(special_tokens)
  return tokenizer


######################### Testing Utils #########################

## Utils for prediction
def category_split(categories, row, left_delim, right_delim):
  categories = categories.replace(' ', '')
  categories = categories.split(sep=right_delim + left_delim)
  if len(categories) != 4 or categories[0][0] != left_delim or categories[3][-1] != right_delim:
    return True
  
  for i in range(3):
    categories[i] += right_delim
    categories[i+1] = left_delim + categories[i+1]

  row.extend(categories)
  return False

def append_to_df(df, row, col_names):
  row_df = pd.DataFrame([row], columns=col_names)
  df = df.append(row_df, ignore_index=True)
  return df

def get_samples_from_actual(df, pred_col, post_ids=None):
  if post_ids is not None:
    actual = df[df.HITId.isin(post_ids)].copy()
    actual = actual[pred_col]
  else:
    actual = df[pred_col].copy()
  
  #actual = actual.sample(n=5)
  actual = categorize_var(actual)
  return actual

def predict_samples(model, tokenizer, actual, pred_col, active_dict, max_length):
  pred = []
  left_delim = tokenizer.sep_token[0]
  right_delim = tokenizer.sep_token[-1]
  
  empty_row = ['' for _ in range(len(pred_col) - 2)]
  bad_output = 0
  bad_categories = 0
  errors = 0
  error_inputs = []

  for i,post in enumerate(list(actual['post'])):
    post_id = actual.iloc[i,0]
    bad_row = [post_id, post] + empty_row
    
    print(post)
    try:
      encoded_post = tokenizer(post, return_tensors='pt')
      output = model.generate(encoded_post['input_ids'], \
                              max_length=max_length, \
                              eos_token_id=tokenizer.eos_token_id)
      output_str = tokenizer.decode(output[0])
    except:
      errors += 1
      error_inputs.append(post)
      pred.append(bad_row)
      continue

    output_list = output_str.split(sep=tokenizer.sep_token)
    if len(output_list) != 5:
      bad_output += 1
      pred.append(bad_row)
      continue

    new_row = [post_id, output_list[0].strip()]
    bad_split = category_split(output_list[1], new_row, left_delim, right_delim)

    if bad_split:
      bad_categories += 1
      pred.append(bad_row)
      continue

    new_row.append(output_list[2].strip())
    new_row.append(output_list[3].strip())
    new_row.append(output_list[4][:-len(tokenizer.eos_token)])
    pred.append(new_row)
  
  pred_df = pd.DataFrame(pred, columns=pred_col)
  print(pred_df)
  actual.to_csv(active_dict['TO ACTUAL'], index=False)
  pred_df.to_csv(active_dict['TO PRED'], index=False)

  print("Errors: ", errors)
  print("Error Tuples: ", error_inputs)
  print("Bad Output: ", bad_output)
  print("Bad Categories: ", bad_categories)

## Utils for Classification Testing

## Utils for Language Generation Testing
def aggregate_and_format(df):
  df = df.replace(np.nan, '', regex=True)

  df.targetMinority = df.targetMinority.str.lower()
  df.targetStereotype = df.targetStereotype.str.lower()
  df.targetMinority = df.targetMinority.str.split('\s*,\s*')
  df.targetStereotype = df.targetStereotype.apply(lambda x: [x])

  df = df.groupby('HITId').agg('sum')
  df.targetMinority = df.targetMinority.apply(lambda x: list(set(x)))
  df.targetStereotype = df.targetStereotype.apply(lambda x: list(set(x)))
  return df

def get_references_and_hypotheses(col_name, actual, pred):
  cmp_grp_target = pd.concat([actual[col_name].rename('actual'), \
                              pred[col_name].rename('pred')], \
                              axis=1)

  cmp_grp_target.pred = cmp_grp_target.pred.replace(np.nan, '', regex=True)
  cmp_grp_target.pred = cmp_grp_target.pred.str.lower()

  references = cmp_grp_target.actual.tolist()
  hypotheses = cmp_grp_target.pred.tolist()
  
  return references, hypotheses

