import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BartTokenizer
from transformers import BartForConditionalGeneration
from seq2seq import *
from datasets import Dataset

# Data Cleaning/Preprocessing Utilities
def clean_post(df):
    df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
    df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
    df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
    df.post = df.post.str.strip()
    return df

def clean_target(df, train=True):
    df.targetStereotype = df.targetStereotype.replace(np.nan, '', regex=True)
    
    if not train:
      # lower case for testing. for training; doesn't matter.
      df.targetStereotype = df.targetStereotype.str.lower()
      df = df.groupby(['HITId', 'post'], as_index=False).agg({'targetStereotype':set})
      df.targetStereotype = df.targetStereotype.apply(lambda x: list(x))
    
    df.rename(columns={"targetStereotype":"target"}, inplace=True)
    return df[['HITId','post','target']]

#def deprocess_labels(labels, pad_id):
#    labels = [
#        [(l if l != -100 else pad_id) for l in label]
#        for label in labels
#    ]
#    return labels

def get_tokenized_data(
    datasets,
    seq2seq_model_name,
    classifier_model_name,
    train=True,
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
      
      if train:
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
 
    if train:
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
    df = clean_target(df, train)

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
        train=train,
    )
    
    return seq2seq_tok, classifier_tok, tokenized

# Model Initialization Function
def init_model(
    model_name,
    join=True,
    join_dropout=0.2,
    classifiers=None,
    train=True,
    use_cuda=True
):
    if join:
      if classifiers is None:
        raise ValueError("You must pass a list of classifiers if initializing a join embedding model")

      model = BartForConditionalGenerationJoinModel.from_pretrained(
                  model_name,
                  join_dropout=join_dropout,
                  classifiers=classifiers,
                  use_cuda=use_cuda
              )
    else:
      model = BartForConditionalGeneration.from_pretrained(model_name)
    
    if use_cuda and torch.cuda.is_available():
      model = model.cuda()
    
    if train:
      model.train()
    else:
      model.eval()
    
    return model

# Testing Utilities
def get_batch(tokenized, i, j, use_cuda=True):
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
    input_ids, attention_mask, classifier_inputs, classifier_attention = get_batch(tokenized, i, j, use_cuda)
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

