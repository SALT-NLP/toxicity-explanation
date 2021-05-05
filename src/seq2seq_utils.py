import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BartTokenizer
from datasets import Dataset

def clean_post(df):
    df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
    df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
    df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
    df.post = df.post.str.strip()
    return df

def clean_target(df):
    df.targetStereotype = df.targetStereotype.replace(np.nan, '', regex=True)
    #df = df.groupby(['HITId', 'post'], as_index=False).agg({'targetStereotype':set})
    #df.targetStereotype = df.targetStereotype.apply(';'.join)
    #df.targetStereotype = df.targetStereotype.str.replace(r'(^;|;$)', '', regex=True)
    df.rename(columns={"targetStereotype":"target"}, inplace=True)
    return df[['HITId','post','target']]

def deprocess_labels(labels, pad_id):
    labels = [
        [(l if l != -100 else pad_id) for l in label]
        for label in labels
    ]
    return labels

def get_tokenized_data(
    datasets,
    seq2seq_model_name,
    classifier_model_name,
    labels=False,
    remove_cols=True,
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

    classifier_tokenized['classifier_inputs'] = classifier_tokenized['input_ids']
    classifier_tokenized['classifier_attention'] = classifier_tokenized['attention_mask']
    del classifier_tokenized['input_ids']
    del classifier_tokenized['attention_mask']
    
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
 
  if remove_cols:
    tokenized = datasets.map(
        tokenize, batched=True,
        num_proc=1,
        remove_columns=['post','target','HITId']
    )
  else:
    tokenized = datasets.map(
        tokenize, batched=True,
        num_proc=1,
    )

  return seq2seq_tok, classifier_tok, tokenized

def tokenize_df(
    from_data,
    to_data,
    seq2seq_model_name,
    classifier_model_name,
    train=True,
    remove_cols=True,
):
    #hitids = [
    #    '3W0XM68YZPPSXA20A826L4NZQHXK11',
    #    '3IYI9285WSUH9T6G8KRE1L6DHMOCJG',
    #    '3ZXV7Q5FJBI14RKKPU0TMNELOFTCFZ',
    #    '3X55NP42EOAPI4DVA4LX5EOVK7XP39',
    #    '33IXYHIZB5CW0VSMXQRHSSKZYQFE2S'
    #]
    #df = df[df.HITId.isin(hitids)]

    df = pd.read_csv(from_data)
    df = clean_post(df)
    df = clean_target(df)

    #df.to_csv(to_data)
    dataset = Dataset.from_pandas(df)
    
    if train:
      datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    else:
      datasets = dataset

    seq2seq_tok, classifier_tok, tokenized = get_tokenized_data(
        datasets,
        seq2seq_model_name,
        classifier_model_name,
        labels=True,
        remove_cols=remove_cols,
    )
    
    return seq2seq_tok, classifier_tok, tokenized

