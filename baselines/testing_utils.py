import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from tqdm import tqdm

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

def predict_samples(model, tokenizer, actual, pred_col, active_test, max_length):
  pred = pd.DataFrame(columns=pred_col)
  left_delim = tokenizer.sep_token[0]
  right_delim = tokenizer.sep_token[-1]
  
  empty_row = ['' for _ in range(len(pred_col) - 2)]
  bad_output = 0
  bad_categories = 0
  errors = 0
  error_inputs = []

  for i,post in enumerate(tqdm(list(actual['post']))):
    post_id = actual.iloc[i,0]
    bad_row = [post_id, post] + empty_row
    
    try:
      encoded_post = tokenizer(post, return_tensors='pt')
      output = model.generate(encoded_post['input_ids'], \
                              max_length=max_length, \
                              eos_token_id=tokenizer.eos_token_id)
      output_str = tokenizer.decode(output[0])
    except:
      errors += 1
      error_inputs.append(post)
      pred = append_to_df(pred, bad_row, pred_col)
      continue

    output_list = output_str.split(sep=tokenizer.sep_token)
    if len(output_list) != 5:
      bad_output += 1
      pred = append_to_df(pred, bad_row, pred_col)
      continue

    new_row = [post_id, output_list[0].strip()]
    bad_split = category_split(output_list[1], new_row, left_delim, right_delim)

    if bad_split:
      bad_categories += 1
      pred = append_to_df(pred, bad_row, pred_col)
      continue

    new_row.append(output_list[2].strip())
    new_row.append(output_list[3].strip())
    new_row.append(output_list[4][:-len(tokenizer.eos_token)])
    pred = append_to_df(pred, new_row, pred_col)

  actual.to_csv(active_test['TO ACTUAL'], index=False)
  pred.to_csv(active_test['TO PRED'], index=False)

  print("Errors: ", errors)
  print("Error Tuples: ", error_inputs)
  print("Bad Output: ", bad_output)
  print("Bad Categories: ", bad_categories)

## Utils for Classification Testing
def f1_score(actual, pred, col, pos, neg):
  tp = pred[(pred[col] == pos) & (pred[col] == actual[col])].shape[0]
  fp = pred[(pred[col] == pos) & (pred[col] != actual[col])].shape[0]
  tn = pred[(pred[col] == neg) & (pred[col] == actual[col])].shape[0]
  fn = pred[(pred[col] == neg) & (pred[col] != actual[col])].shape[0]
  
  if tp + fp == 0:
    precision = 0
  else:
    precision = tp / float(tp + fp)

  if tp + fn == 0:
    recall = 0
  else:
    recall = tp / float(tp + fn)
  
  if precision + recall == 0:
    f1 = 0
  else:
    f1 = 2 * ((precision*recall) / (precision + recall))
  
  return precision, recall, f1

def accuracy(actual, pred, col):
  match = pred[pred[col] == actual[col]].shape[0]
  total = pred.shape[0]
  return match / float(total)

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

def get_bleu_score(references, hypotheses):
  tokenized_hypotheses = list(map(str.split, hypotheses))
  tokenized_references = list(map(lambda s: list(map(str.split, s)), references))
  
  bleu = np.empty((len(hypotheses), 2))
  for i,hyp in enumerate(hypotheses):
    bleu_ref = np.empty(len(references[i]))
    for j,ref in enumerate(references[i]):
      if len(ref) == 0 and len(hyp) == 0:
        bleu_ref[j] = 1.0
      elif len(ref) == 0 and len(hyp) != 0:
        bleu_ref[j] = 0.0
      elif len(ref) != 0 and len(hyp) == 0:
        bleu_ref[j] = 0.0
      else:
        bleu_ref[j] = sentence_bleu([ref], hyp, weights=(0.5, 0.5))
    bleu[i] = [np.max(bleu_ref), np.average(bleu_ref)]
  
  return np.average(bleu, axis=0)

def get_rouge_scores(references, hypotheses):
  rouge_scores = np.empty((len(hypotheses), 2, 3))
  rouge = Rouge(metrics=['rouge-l'])

  for i, hyp in enumerate(hypotheses):
    ref_scores = np.empty((len(references[i]), 3))
    for j, ref in enumerate(references[i]):
      if len(ref) == 0 and len(hyp) == 0:
        scores = [{'rouge-l': {'f': 1.0, 'p': 1.0, 'r': 1.0}}]
      elif len(ref) == 0 and len(hyp) != 0:
        scores = [{'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
      elif len(ref) != 0 and len(hyp) == 0:
        scores = [{'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
      else:
        scores = rouge.get_scores(hyp, ref)
      ref_scores[j, 0] = scores[0]['rouge-l']['p']
      ref_scores[j, 1] = scores[0]['rouge-l']['r']

      if ref_scores[j, 0] + ref_scores[j, 1] == 0.0:
        ref_scores[j, 2] = 0.0
      elif np.isnan(ref_scores[j, 0]):
        ref_scores[j, 2] = np.nan
      else:
        ref_scores[j, 2] = 2 * ((ref_scores[j, 0] * ref_scores[j, 1]) / \
                                (ref_scores[j, 0] + ref_scores[j, 1]))

    max_j = np.argmax(ref_scores, axis=0)[2]
    rouge_scores[i,0,:] = ref_scores[max_j]
    rouge_scores[i,1,:] = np.average(ref_scores, axis=0)
  return np.average(rouge_scores, axis=0)

def get_bert_score(bert_scores, hypotheses, references):
  for i,_ in enumerate(hypotheses):
    if len(hypotheses[i]) == 0:
      if len(references[i]) == 1:
        if len(references[i][0]) == 0:
          bert_scores['precision'][i] = 1.0
          bert_scores['recall'][i] = 1.0
          bert_scores['f1'][i] = 1.0
        else:
          bert_scores['precision'][i] = 0.0
          bert_scores['recall'][i] = 0.0
          bert_scores['f1'][i] = 0.0
      else:
        bert_scores['precision'][i] = 0.0
        bert_scores['recall'][i] = 0.0
        bert_scores['f1'][i] = 0.0
    elif len(references[i]) == 1:
      if len(references[i][0]) == 0:
        bert_scores['precision'][i] = 0.0
        bert_scores['recall'][i] = 0.0
        bert_scores['f1'][i] = 0.0

  precision = np.average(bert_scores['precision'])
  recall = np.average(bert_scores['recall'])
  f1 = 2 * (precision * recall) / (precision + recall)
  return precision, recall, f1

