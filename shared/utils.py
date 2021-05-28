import math
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge

# Classification Utils
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


# Language Generation Utils
def get_bleu_score(references, hypotheses):
    #tokenized_hypotheses = list(map(str.split, hypotheses))
    #tokenized_references = list(map(lambda s: list(map(str.split, s)), references))
    
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

# Model Training Utils
def get_step_variables(
    num_rows,
    num_epochs,
    batch_size,
    warmup_div=None,
    warmup_one_epoch=True
):
  if num_epochs == 1:
    one_epoch_steps = math.ceil(num_rows / batch_size) // 2
    save_steps = one_epoch_steps * 2
    eval_steps = (save_steps * 5.0) // 100
  else:
    one_epoch_steps = math.ceil(num_rows / batch_size)
    save_steps = (one_epoch_steps * num_epochs) // 2
    eval_steps = (one_epoch_steps * num_epochs * 5.0) // 100
  
  if warmup_one_epoch:
    warmup_steps = one_epoch_steps
  else:
    warmup_steps = (one_epoch_steps * num_epochs) // warmup_div
  
  return warmup_steps, save_steps, eval_steps

