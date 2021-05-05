import pandas as pd
import numpy as np
import math
import pickle
from seq2seq_utils import *
from seq2seq import BartForConditionalGenerationJoinModel
from torch import nn, torch
from datasets import Dataset,load_metric
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from train import *

# Useful constants
CLASSIFIER_MODEL_NAME = 'bert-base-uncased'
CLASSIFIERS = ['./classification/model/whoTarget/checkpoint-1280/']
#CLASSIFIERS = ['./classification/model/offensiveYN/checkpoint-898/',
#               './classification/model/intentYN/checkpoint-898/',
#               './classification/model/sexYN/checkpoint-898/',
#               './classification/model/whoTarget/checkpoint-1280/']

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

SEQ2SEQ_BART_BASE = './model/bart_base_checkpoint-17970/'
SEQ2SEQ_PICKLE_BASE = './data/bart_base_checkpoint-17970.pickle'

SEQ2SEQ_BART_JOIN = './model/bart_join_pretrain_zeros_group_checkpoint-21560/'
SEQ2SEQ_PICKLE_JOIN = './data/bart_join_pretrain_zeros_group_checkpoint-21560.pickle'

DATA_DIR = '../data/'
FROM_DATA = DATA_DIR + 'SBIC.v2.dev.csv'
TO_DATA = 'data/clean_dev_df.csv'

def get_batch(tokenized, i, j):
  if i == j:
    input_ids = torch.tensor([tokenized['input_ids'][i]])
    attention_mask = torch.tensor([tokenized['attention_mask'][i]])
    classifier_inputs = torch.tensor([tokenized['classifier_inputs'][i]])
    classifier_attention = torch.tensor([tokenized['classifier_attention'][i]])
  elif i < j:
    input_ids = torch.tensor(tokenized['input_ids'][i:j])
    attention_mask = torch.tensor(tokenized['attention_mask'][i:j])
    classifier_inputs = torch.tensor(tokenized['classifier_inputs'][i:j])
    classifier_attention = torch.tensor(tokenized['classifier_attention'][i:j])
  else:
    raise ValueError("Pass value i <= j")
  
  if use_cuda and torch.cuda.is_available():
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    classifier_inputs = classifier_inputs.cuda()
    classifier_attention = classifier_attention.cuda()
  
  return input_ids, attention_mask, classifier_inputs, classifier_attention

def generate_batch(tokenized, tokenizer, model, i, j, use_cuda=True):
  input_ids, attention_mask, classifier_inputs, classifier_attention = get_batch(tokenized, i, j)
  num_beams = 10

  if isinstance(model, BartForConditionalGenerationJoinModel):
    encoder_outputs = model.encoder_enrichment_forward(
        input_ids,
        classifier_inputs,
        attention_mask,
        classifier_attention,
        return_dict=True,
    )
  else:
    encoder_outputs = model.get_encoder()(
        input_ids,
        attention_mask,
        return_dict=True,
    )
  
  output_ids = model.generate(input_ids, num_beams=num_beams, length_penalty=5.0)
  input_strs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
  output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  return input_strs, output_strs

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

def generate_stereotypes(tokenized, seq2seq_tok, model, pickle_file, use_cuda=True):
  batch_size = 10
  results = [[],[]]
  num_batches = math.ceil(tokenized.num_rows // batch_size)

  for batch in tqdm(range(num_batches)):
    i = batch * batch_size
    j = min(tokenized.num_rows, i + batch_size)
    
    _, output_strs = generate_batch(tokenized, seq2seq_tok, model, i, j, use_cuda=use_cuda)
    #targets = [string.split(';') for string in tokenized['target'][i:j]]
    targets = [[string] for string in tokenized['target'][i:j]]
    results[0].extend(targets)
    results[1].extend(output_strs)
  pickle.dump(results, open(pickle_file, 'wb'))

def generate_scores(pickle_file):
  results = pickle.load(open(pickle_file, 'rb'))
  references = results[0]
  hypotheses = results[1]

  bleu_score_max, bleu_score_avg = get_bleu_score(references, hypotheses)
  rouge_scores_max, rouge_scores_avg = get_rouge_scores(references, hypotheses)

  metric = load_metric('bertscore')
  bert_scores = metric.compute(predictions=hypotheses, references=references, lang='en')
  bert_score = get_bert_score(bert_scores, hypotheses, references)

  print("Bleu Score (Avg): ", bleu_score_avg)
  print("Bleu Score (Max): ", bleu_score_max)
  print("Rouge Score (Avg) (Precision, Recall, F1): ", rouge_scores_avg)
  print("Rouge Score (Max) (Precision, Recall, F1): ", rouge_scores_max)
  print('BERT Score (Max) (Precision, Recall, F1): ', bert_score)

if __name__ == '__main__':
  print('preparing and tokenizing data ...')
  seq2seq_tok, classifier_tok, tokenized = tokenize_df(
      FROM_DATA,
      TO_DATA,
      SEQ2SEQ_MODEL_NAME,
      CLASSIFIER_MODEL_NAME,
      train=False,
      remove_cols=False,
  )
  use_cuda = True

  print('initializing model ...')
  join_model = init_model(SEQ2SEQ_BART_JOIN, join=True, train=False, use_cuda=use_cuda)
  pickle_file = SEQ2SEQ_PICKLE_JOIN
  
  #base_model = init_model(SEQ2SEQ_BART_BASE, join=False, train=False, use_cuda=use_cuda)
  #pickle_file = SEQ2SEQ_PICKLE_BASE
  
  # Run Tests
  print('running tests ...')
  generate_stereotypes(tokenized, seq2seq_tok, join_model, pickle_file, use_cuda=use_cuda)

  print("generating scores ...")
  generate_scores(pickle_file)

  ### Print Model Output ###
  """
  print("Base Model")
  input_strs, output_strs = generate_batch(tokenized, seq2seq_tok, base_model, 0, 5, use_cuda=use_cuda)
  for i in range(len(input_strs)):
    print('Input Sentence: ', input_strs[i])
    print('Output Stereotype: ', output_strs[i])
    print('\n')
  
  
  print("Join Model")
  input_strs, output_strs = generate_batch(tokenized, seq2seq_tok, join_model, 0, 5, use_cuda=use_cuda)
  for i in range(len(input_strs)):
    print('Input Sentence: ', input_strs[i])
    print('Output Stereotype: ', output_strs[i])
    print('\n')
  """

