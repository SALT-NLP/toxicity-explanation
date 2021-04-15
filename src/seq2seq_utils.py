import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BartTokenizer

def clean_post(df):
    df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
    df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
    df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
    df.post = df.post.str.strip()
    return df

def clean_target(df):
    df.targetStereotype = df.targetStereotype.replace(np.nan, '', regex=True)
    df = df.groupby(['HITId', 'post'], as_index=False).agg({'targetStereotype':set})
    df.targetStereotype = df.targetStereotype.apply(';'.join)
    df.targetStereotype = df.targetStereotype.str.replace(r'(^;|;$)', '', regex=True)
    df.rename(columns={"targetStereotype":"target"}, inplace=True)
    return df

def get_tokenized_data(
    datasets,
    seq2seq_model_name,
    classifier_model_name,
    labels=False
):
  def process_labels(target_tokenized):
    target_tokenized['labels'] = [
        [(l if l != seq2seq_tok.pad_token_id else -100) for l in label]
        for label in target_tokenized['input_ids']
    ]
    
    del target_tokenized['input_ids']
    del target_tokenized['attention_mask']
  
  def tokenize(examples):
    seq2seq_tokenized = seq2seq_tok(
        examples['post'],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
  
    classifier_tokenized = classifier_tok(
        examples['post'],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
  
    seq2seq_tokenized['input_ids'] = seq2seq_tokenized['input_ids']
    seq2seq_tokenized['attention_mask'] = seq2seq_tokenized['attention_mask']
    
    classifier_tokenized['classifier_inputs'] = classifier_tokenized['input_ids']
    classifier_tokenized['classifier_attention'] = classifier_tokenized['attention_mask']

    if labels:
      with seq2seq_tok.as_target_tokenizer():
        target_tokenized = seq2seq_tok(
          examples['target'],
          padding="max_length",
          truncation=True,
          max_length=128,
        )
      process_labels(target_tokenized)  
      return {**seq2seq_tokenized, **classifier_tokenized, **target_tokenized}
    return {**seq2seq_tokenized, **classifier_tokenized}
  
  #### get_tokenized_data function body
  seq2seq_tok = BartTokenizer.from_pretrained(seq2seq_model_name)
  classifier_tok = BertTokenizer.from_pretrained(classifier_model_name)
 
  tokenized = datasets.map(
      tokenize, batched=True,
      num_proc=4,
      remove_columns=['post','target','HITId']
  )

  return seq2seq_tok, classifier_tok, tokenized

