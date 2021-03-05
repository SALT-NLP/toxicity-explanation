import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

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

def predict_samples(model, tokenizer, actual, pred_col, active_test):
  pred = pd.DataFrame(columns=pred_col)
  left_delim = tokenizer.sep_token[0]
  right_delim = tokenizer.sep_token[-1]
  
  empty_row = ['' for _ in range(len(pred_col) - 2)]
  bad_output = 0
  bad_categories = 0
  errors = 0
  error_inputs = []

  for i,post in enumerate(list(actual['post'])):
    if not(i % 100):
      print(i)
   
    post_id = actual.iloc[i,0]
    bad_row = [post_id, post] + empty_row
    
    try:
      encoded_post = tokenizer(post, return_tensors='pt')
      output = model.generate(encoded_post['input_ids'], \
                              max_length=150, \
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
  #empty_str_list = lambda x: (len(x) > 1) | (x[0] != "")

  cmp_grp_target.pred = cmp_grp_target.pred.replace(np.nan, '', regex=True)
  cmp_grp_target.pred = cmp_grp_target.pred.str.lower()
  #cmp_grp_target = cmp_grp_target[cmp_grp_target['actual'].map(empty_str_list)]

  references = cmp_grp_target.actual.tolist()
  hypotheses = cmp_grp_target.pred.tolist()
  
  return references, hypotheses

def get_bleu_score(references, hypotheses):
  tokenized_hypotheses = list(map(str.split, hypotheses))
  tokenized_references = list(map(lambda s: list(map(str.split, s)), references))
  return corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))

def get_rouge_scores(references, hypotheses):
  rouge_scores = np.empty((len(hypotheses), 3))
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  for i, hyp in enumerate(hypotheses):
    f1_scores = np.empty((len(references[i]), 3))
    for j, ref in enumerate(references[i]):
      scores = scorer.score(ref, hyp)
      f1_scores[j] = list(scores['rougeL'])
    max_j = np.argmax(f1_scores, axis=0)[2]
    rouge_scores[i] = f1_scores[max_j]
  return np.average(rouge_scores, axis=0)

