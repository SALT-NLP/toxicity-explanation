import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
from transformers import BertTokenizer, BartTokenizer
from transformers import BartForConditionalGeneration
from seq2seq import *
from datasets import Dataset

################################ Data Cleaning/Preprocessing ################################

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

# Reads and cleans the CSV file passed in.
# Returns a dataset object.
def read_and_clean_csv(
    from_data,
    to_data,
    train=True,
):
    df = pd.read_csv(from_data)
    df = clean_post(df)
    df = clean_target(df, train)
    df.to_csv(to_data)

    dataset = Dataset.from_pandas(df)
    return dataset

################################ Tokenization ################################

# Tokenizes for BERT Classifier
def tokenize_classifier_df(
    dataset,
    classifier_tok_name,
):
    ## Local Function for tokenizing input.
    def tokenize(examples):
      classifier_tokenized = classifier_tok(
          examples['post'],
          padding="max_length",
          truncation=True,
          max_length=128,
      )
      return classifier_tokenized

    classifier_tok = BertTokenizer.from_pretrained(classifier_tok_name)
    tokenized = dataset.map(
        tokenize, batched=True,
        num_proc=1,
        remove_columns=['post','target','HITId']
    )
    return tokenized

# Tokenizes the dataset for encoder-decoder BART model
def tokenize_bart_df(
    dataset,
    seq2seq_tok_name,
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
    
      if train:
        with seq2seq_tok.as_target_tokenizer():
          target_tokenized = seq2seq_tok(
              examples['target'],
              padding="max_length",
              truncation=True,
              max_length=128,
          )
        process_labels(target_tokenized)
        return {**seq2seq_tokenized, **target_tokenized}
      return seq2seq_tokenized
    
    #### get_tokenized_data function body
    seq2seq_tok = BartTokenizer.from_pretrained(seq2seq_tok_name)
 
    if train:
      tokenized = dataset.map(
          tokenize, batched=True,
          num_proc=1,
          remove_columns=['post','target','HITId']
      )
    else:
      tokenized = dataset.map(
          tokenize, batched=True,
          num_proc=1,
      )

    return seq2seq_tok, tokenized

# Takenizes a dataset, and a list of classifiers and computes the attentions
# for each classifier.
def get_classifier_attention(dataset, classifier_tok_name, classifiers, use_cuda=True):
    inputs = tokenize_classifier_df(dataset, classifier_tok_name)
    
    # Need to move to PyTorch tensor since huggingface tokenizer does
    # not always do it.
    input_ids = torch.tensor(inputs['input_ids'])
    attention_mask = torch.tensor(inputs['attention_mask'])
    if use_cuda and torch.cuda.is_available():
      input_ids = input_ids.cuda()
      attention_mask = attention_mask.cuda()
    
    batch_size = 20
    num_rows = input_ids.shape[0]
    num_batches = math.ceil(num_rows / batch_size)

    attentions = []
    for k,classifier in enumerate(classifiers):
      print("Running Classifier: ", classifier)
      model = BertForSequenceClassification.from_pretrained(classifier)
      if use_cuda and torch.cuda.is_available():
        model = model.cuda()
      
      attentions.append([])
      for batch in tqdm(range(num_batches)):
        i = batch * batch_size
        j = min(i + batch_size, num_rows)

        output = model(input_ids[i:j], attention_mask[i:j], output_attentions=True)
        attn_layers = output.attentions[-1].mean(dim=2)
        attentions[k].extend(attn_layers.tolist())
    attentions = np.stack(attentions, axis=1)
    return attentions

# Maps column to a dataset. There should be an column value for each
# entry in the dataset.
def map_column_to_dataset(dataset, column, column_name):
    # Internal function for dataset mapping
    def update_example(example, index):
      example[column_name] = column[index].tolist()
      return example
    
    dataset = dataset.map(update_example, with_indices=True)
    return dataset


################################ Model Initialization ################################

# Model Initialization Function
def init_model(
    model_name,
    join=True,
    join_dropout=0.2,
    num_classifiers=4,
    num_classification_heads=12,
    train=True,
    use_cuda=True
):
    if join:
      model = BartForConditionalGenerationJoinModel.from_pretrained(
                  model_name,
                  join_dropout=join_dropout,
                  num_classifiers=num_classifiers,
                  num_classification_heads=num_classification_heads,
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

################################ Testing Utilities ################################

# Returns a batch from the tokenized dataset between i-th and j-th rows
def get_batch(tokenized, i, j, join=True, use_cuda=True):
    # Need to move to pytorch tensor since huggingface does not
    # always do it
    if i == j:
      input_ids = torch.tensor([tokenized['input_ids'][i]])
      attention_mask = torch.tensor([tokenized['attention_mask'][i]])
      classifier_attention = torch.tensor([tokenized['classifier_attention'][i]]) if join else None
    elif i < j:
      input_ids = torch.tensor(tokenized['input_ids'][i:j])
      attention_mask = torch.tensor(tokenized['attention_mask'][i:j])
      classifier_attention = torch.tensor(tokenized['classifier_attention'][i:j]) if join else None
    else:
      raise ValueError("Pass value i <= j")
    
    if use_cuda and torch.cuda.is_available():
      input_ids = input_ids.cuda()
      attention_mask = attention_mask.cuda()
      classifier_attention = classifier_attention.cuda() if join else None
    
    return input_ids, attention_mask, classifier_attention

def generate_batch(tokenized, tokenizer, model, i, j, use_cuda=True):
    join = isinstance(model, BartForConditionalGenerationJoinModel)
    input_ids, attention_mask, classifier_attention = get_batch(tokenized, i, j, join, use_cuda)
    num_beams = 10

    if join:
      encoder_outputs = model.encoder_enrichment_forward(
          input_ids,
          classifier_attention,
          attention_mask=attention_mask,
          return_dict=True,
      )
    else:
      encoder_outputs = model.get_encoder()(
          input_ids,
          attention_mask,
          return_dict=True,
      )
    
    model_kwargs = {'encoder_outputs': encoder_outputs}
    output_ids = model.generate(input_ids, num_beams=num_beams, length_penalty=5.0, **model_kwargs)
    
    input_strs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return input_strs, output_strs

