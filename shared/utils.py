import math
import torch
import os
import pickle
import random
import csv
import collections
import numpy as np
import pandas as pd
from datasets import load_metric, Dataset
from transformers import BartTokenizer, Trainer, TrainingArguments
from transformers.tokenization_utils_base import BatchEncoding
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from multi_view_trainer import MultiViewTrainer

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
def get_bleu_score(references, hypotheses, return_all_scores=False):
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
    
    if return_all_scores:
      return bleu
    else:
      return np.average(bleu, axis=0)

def get_rouge_scores(references, hypotheses, return_all_scores=False):
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
    if return_all_scores:
      return rouge_scores
    else:
      return np.average(rouge_scores, axis=0)

def get_bert_score(bert_scores, hypotheses, references, return_all_scores=False):
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
    f1 = np.average(bert_scores['f1'])
    #f1 = 2 * (precision * recall) / (precision + recall)
    if return_all_scores:
      return bert_scores
    else:
      return precision, recall, f1

# Testing Utils
def generate_stereotypes(
    batch_iter,
    tokenizer,
    model,
    model_enc_func,
    results_cols=['HITId', 'post'],
    pickle_file=None,
    num_views=6,
    view_model=False
):
    if os.path.isfile(pickle_file):
      results = pickle.load(open(pickle_file, 'rb'))
      return results

    results = [[] for _ in range(len(results_cols) + 1)]
    
    for batch in tqdm(batch_iter):
      if view_model:
        _, output_strs, view_attention = generate_stereotype(
          batch,
          tokenizer,
          model,
          model_enc_func,
          batch_iter.torch_cols,
          num_views,
          view_model
        )
      else:
        _, output_strs = generate_stereotype(
          batch,
          tokenizer,
          model,
          model_enc_func,
          batch_iter.torch_cols,
          num_views,
          view_model,
        )
      for i,col in enumerate(results_cols):
        if view_model:
          col_result = []
          if col == 'view_attention':
            results[i].extend(view_attention.cpu().tolist())
          else:
            for j in range(0, len(batch[col]), num_views):
              if col == 'post':
                col_result.append(';'.join(batch[col][j:j+num_views]))
              else:
                col_result.append(batch[col][j])
            results[i].extend(col_result)
        else:
          results[i].extend(batch[col])
      results[i + 1].extend([output_str.lower() for output_str in output_strs])

    df_dict = {}
    for i,col in enumerate(results_cols):
      df_dict[col] = results[i]
    df_dict['prediction'] = results[-1]
    
    results = pd.DataFrame(df_dict)
    if pickle_file is not None:
      pickle.dump(results, open(pickle_file, 'wb'))
    
    return results

def generate_stereotype(
    batch,
    tokenizer,
    model,
    model_enc_func,
    encoder_input_cols=['input_ids','attention_mask'],
    num_views=6,
    view_model=False
):
    num_beams = 10
    enc_input = []
    
    for col in encoder_input_cols:
      enc_input.append(batch[col])
    
    encoder_outputs = model_enc_func(*enc_input, return_dict=True)
    if view_model:
      model_kwargs = {'encoder_outputs_by_view': encoder_outputs}
    else:
      model_kwargs = {'encoder_outputs': encoder_outputs}
    output_ids = model.generate(batch['input_ids'], num_beams=num_beams, length_penalty=5.0, num_views=num_views, **model_kwargs)

    input_strs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    if view_model:
      return input_strs, output_strs, encoder_outputs[1]
    else:
      return input_strs, output_strs

def generate_scores(pickle_file, save_results_to_csv=True, generation_seed=756, num_results=200, save_file='results.csv', view_model=False):
    results = pickle.load(open(pickle_file, 'rb'))
    references = results['target'].tolist()
    hypotheses = results['prediction'].tolist()
    
    if not(save_results_to_csv):
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
      
      if view_model:
        view_attention = results['view_attention'].tolist()
        print('View Attention: ', np.average(view_attention, axis=0))
    else:
      indices = list(range(len(references)))
      random.seed(generation_seed)
      random.shuffle(indices)

      col_names = ['HITId', 'post', 'target','prediction']
      results_csv = []
      for idx in indices[:num_results]:
        results_csv.append([
          results[col_names[0]][idx],
          results[col_names[1]][idx].replace('\n', ' '),
          ', '.join(results[col_names[2]][idx]),
          results[col_names[3]][idx],
        ])
      
      with open(save_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter='|')
        csv_writer.writerow(col_names)
        csv_writer.writerows(results_csv)
        

# Minibatch Iterator Class
class MinibatchIterator:
    # torch_cols must be in the same order that the model accepts the arguments.
    def __init__(self, data, tokenizer, batch_size=16, max_length=1024, use_cuda=True, torch_cols=['input_ids', 'attention_mask']):
      self.data = data
      self.tokenizer = tokenizer
      self.batch_size = batch_size
      self.max_length = max_length
      self.use_cuda = use_cuda
      self.torch_cols = torch_cols
      self.current_pos = 0

    def __iter__(self):
      return self

    def __next__(self):
      if self.current_pos == self.data.num_rows:
        raise StopIteration

      self.next_pos = min(self.data.num_rows, self.current_pos + self.batch_size)
      next_batch = self.data[self.current_pos:self.next_pos]
      next_batch = self.tokenizer.pad(next_batch, padding=True, max_length=self.max_length)
      
      for col in self.torch_cols:
        next_batch[col] = torch.tensor(next_batch[col])
        if self.use_cuda:
          next_batch[col] = next_batch[col].cuda()
      
      self.current_pos = self.next_pos
      return next_batch
  
def get_hitids(hitid_file):
    return set(pd.read_csv(hitid_file).iloc[:,0])

def get_file_name(file_path, remove_extension=True):
    file_name = os.path.basename(os.path.normpath(file_path))
    if remove_extension:
      return os.path.splitext(file_name)[0]
    return file_name

# Model Training Utils

def get_step_variables(
    num_rows,
    num_epochs,
    batch_size,
    warmup_div=None,
    warmup_one_epoch=True,
    eval_percent=5.0
):
    if num_epochs == 1:
      one_epoch_steps = math.ceil(num_rows / batch_size) // 2
      save_steps = one_epoch_steps * 2
      eval_steps = (save_steps * eval_percent) // 100
    else:
      one_epoch_steps = math.ceil(num_rows / batch_size)
      save_steps = (one_epoch_steps * num_epochs) // 2
      eval_steps = (one_epoch_steps * num_epochs * 5.0) // 100
    
    if warmup_one_epoch:
      warmup_steps = one_epoch_steps
    else:
      warmup_steps = (one_epoch_steps * num_epochs) // warmup_div
    
    return warmup_steps, save_steps, eval_steps

def train(
    model,
    datasets,
    num_epochs=3.0,
    learning_rate=5e-6,
    batch_size=4,
    eval_percent=10.0,
    data_collator=None,
    view_model=False,
    num_views=6,
):
    num_rows = datasets['train'].num_rows
    warmup_steps, save_steps, eval_steps = get_step_variables(
        num_rows, num_epochs, batch_size, eval_percent=eval_percent
    )
    
    print("Linear Warm Up: ", warmup_steps)
    print("Save Steps: ", save_steps)
    print("Eval Steps: ", eval_steps)

    training_args = TrainingArguments(
        output_dir = 'model',
        evaluation_strategy = 'steps',
        eval_steps = eval_steps,
        logging_steps = eval_steps,
        save_steps = save_steps,
        save_total_limit = 1,
        warmup_steps = warmup_steps,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epochs,
    )
    
    if view_model:
      trainer = MultiViewTrainer(
          model=model,
          args=training_args,
          train_dataset=datasets["train"],
          eval_dataset=datasets["test"],
          data_collator=data_collator,
          num_views=num_views
      )
    else:
      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=datasets["train"],
          eval_dataset=datasets["test"],
          data_collator=data_collator,
      )
    trainer.train()

# Warning Overwrite
def custom_warning(msg, *args, **kwargs):
    return 'WARNING: ' + str(msg) + '\n'

# Dataset Utils

# Data Cleaning/Preprocessing

def clean_post(df):
    df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
    df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
    df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
    df.post = df.post.str.strip()
    return df

def clean_target(df, target_col='targetStereotype', train=True):
    df[target_col] = df[target_col].replace(np.nan, '', regex=True)
    
    if not train:
      # lower case for testing. for training; doesn't matter.
      df[target_col] = df[target_col].str.lower()
      df = df.groupby(['HITId', 'post'], as_index=False).agg({target_col:set})
      df[target_col] = df[target_col].apply(lambda x: list(x))
    
    df.rename(columns={target_col:'target'}, inplace=True)
    return df[['HITId','post','target']]

# Maps column to a dataset. There should be an column value for each
# entry in the dataset.
def map_column_to_dataset(dataset, column, column_name):
    # Internal function for dataset mapping
    def update_example(example, index):
      example[column_name] = column[index].tolist()
      return example
    
    dataset = dataset.map(update_example, with_indices=True)
    return dataset

# Tokenization Utils
# Tokenizes the dataset for encoder-decoder BART model
def tokenize_bart_df(
    dataset,
    seq2seq_tok_name,
    train=True,
    padding=True,
    max_length=128,
    target_max_length=128,
    special_tokens=[],
    view_model=False,
    num_views=6,
):
    def process_labels(target_tokenized):
      target_tokenized['labels'] = [
          [(l if l != seq2seq_tok.pad_token_id else -100) for l in label]
          for label in target_tokenized['input_ids']
      ]
      
      del target_tokenized['input_ids']
      del target_tokenized['attention_mask']
    
    def tokenize(examples):
      pad_examples = "max_length" if padding else False
      
      seq2seq_tokenized = seq2seq_tok(
          examples['post'],
          padding=pad_examples,
          truncation=True,
          max_length=max_length,
      )

      if train:
        with seq2seq_tok.as_target_tokenizer():
          target_tokenized = seq2seq_tok(
              examples['target'],
              padding=pad_examples,
              truncation=True,
              max_length=max_length,
          )
        process_labels(target_tokenized)
        return {**seq2seq_tokenized, **target_tokenized}
      return seq2seq_tokenized
    
    #### get_tokenized_data function body
    seq2seq_tok = BartTokenizer.from_pretrained(seq2seq_tok_name, additional_special_tokens=special_tokens)
 
    if train:
      remove_cols = ['post', 'target', 'HITId']
      tokenized = dataset.map(
          tokenize, batched=True,
          num_proc=1,
          remove_columns=remove_cols
      )
    else:
      tokenized = dataset.map(
          tokenize, batched=True,
          num_proc=1,
      )

    return seq2seq_tok, tokenized

# Dataset Statistics Helpers
def generate_length_distribution(dataset, partition=20):
    input_tokens = dataset['input_ids']
    lengths = list(map(len, input_tokens))
    lengths.sort()

    ntiles = []
    for i in range(partition):
      ntiles.append(float(i / partition) * 100)
    
    print('Percentile: ', np.percentile(lengths, ntiles))

