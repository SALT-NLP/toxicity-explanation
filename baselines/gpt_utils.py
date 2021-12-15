"""
This file contains data cleaning functions for the SBIC corpus

See the SBIC Corpus here:
  https://homes.cs.washington.edu/~msap/social-bias-frames/DATASTATEMENT.html
"""
import pickle
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset, load_metric
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from utils import *

# String Separator
SEP = '[SEP]'
BOS = '[STR]'
EOS = '[END]'
PAD = '[PAD]'

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
  df = clean_post(df)
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
  
  actual = aggregate_and_format(actual)
  actual = categorize_var(actual)
  return actual

def predict_samples(model, tokenizer, actual, pred_col, active_dict, max_length, num_beams):
  pred = []
  left_delim = tokenizer.sep_token[0]
  right_delim = tokenizer.sep_token[-1]
  
  empty_row = ['' for _ in range(len(pred_col) - 2)]
  bad_output = 0
  bad_categories = 0
  errors = 0
  error_inputs = []
  
  for i,post in enumerate(tqdm(list(actual.post))):
    post_id = actual.iloc[i,0]
    bad_row = [post_id, post] + empty_row
    
    try:
      encoded_post = tokenizer(post, return_tensors='pt')
      output = model.generate(
          encoded_post['input_ids'],
          max_length=max_length,
          eos_token_id=tokenizer.eos_token_id,
          num_beams=num_beams
      )
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
  pickle.dump(actual, open(active_dict['TO ACTUAL'], 'wb'))
  pickle.dump(pred_df, open(active_dict['TO PRED'], 'wb'))

  print("Errors: ", errors)
  print("Error Tuples: ", error_inputs)
  print("Bad Output: ", bad_output)
  print("Bad Categories: ", bad_categories)

## Utils for Classification Testing
def get_and_print_f1_scores(actual, pred):
  print("Category: (Precision, Recall, F1)")
  print('Offensive: ', f1_score(actual, pred, 'offensiveYN', OFFY, OFFN))
  print('Intent: ', f1_score(actual, pred, 'intentYN', INTY, INTN))
  print('Lewd: ', f1_score(actual, pred, 'sexYN', LEWDY, LEWDN))
  print('Group Targeted: ', f1_score(actual, pred, 'whoTarget', GRPY, GRPN))
  print('In Group: ', f1_score(actual, pred, 'speakerMinorityYN', INGY, INGN))

## Utils for Language Generation Testing
def get_and_print_lang_gen_scores(df, actual, pred):
  references_tm, hypotheses_tm = get_references_and_hypotheses('targetMinority', actual, pred)
  bleu_score_tm_max, bleu_score_tm_avg = get_bleu_score(references_tm, hypotheses_tm)
  rouge_scores_tm_max, rouge_scores_tm_avg = get_rouge_scores(references_tm, hypotheses_tm)
  
  references_ts, hypotheses_ts = get_references_and_hypotheses('targetStereotype', actual, pred)
  bleu_score_ts_max, bleu_score_ts_avg = get_bleu_score(references_ts, hypotheses_ts)
  rouge_scores_ts_max, rouge_scores_ts_avg = get_rouge_scores(references_ts, hypotheses_ts)
    
  metric = load_metric('bertscore')
  bert_scores_tm = metric.compute(predictions=hypotheses_tm, references=references_tm, lang='en')
  bert_score_tm = get_bert_score(bert_scores_tm, hypotheses_tm, references_tm)
  
  bert_scores_ts = metric.compute(predictions=hypotheses_ts, references=references_ts, lang='en')
  bert_score_ts = get_bert_score(bert_scores_ts, hypotheses_ts, references_ts)
  
  print("Target Minority Scores: ")
  print("Bleu Score (Avg): ", bleu_score_tm_avg)
  print("Bleu Score (Max): ", bleu_score_tm_max)
  print("Rouge Score (Avg) (Precision, Recall, F1): ", rouge_scores_tm_avg)
  print("Rouge Score (Max) (Precision, Recall, F1): ", rouge_scores_tm_max)
  print('BERT Score (Max) (Precision, Recall, F1): ', bert_score_tm)
  
  print("Implied Stereotype Scores: ")
  print("Bleu Score (Avg): ", bleu_score_ts_avg)
  print("Bleu Score (Max): ", bleu_score_ts_max)
  print("Rouge Score (Avg) (Precision, Recall, F1): ", rouge_scores_ts_avg)
  print("Rouge Score (Max) (Precision, Recall, F1): ", rouge_scores_ts_max)
  print('BERT Score (Max) (Precision, Recall, F1): ', bert_score_ts)

def aggregate_and_format(df):
  df = df.fillna({
      'sexYN': 0.0, 'offensiveYN': 0.0, 'intentYN': 0.0, 'whoTarget': 0.0,
      'targetMinority': '', 'targetStereotype': '', 'speakerMinorityYN': 0.0
  })
  
  df.targetMinority = df.targetMinority.str.lower()
  df.targetStereotype = df.targetStereotype.str.lower()
  df.targetMinority = df.targetMinority.str.split('\s*,\s*')
  df.targetStereotype = df.targetStereotype.apply(lambda x: [x])

  df = df.groupby(['HITId','post']).agg({
    'sexYN': 'mean', 'offensiveYN': 'mean', 'intentYN': 'mean', 'whoTarget': 'mean',
    'targetMinority': 'sum', 'targetStereotype': 'sum', 'speakerMinorityYN': 'mean'
  }).reset_index()
  
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

