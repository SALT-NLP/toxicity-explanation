import sys
sys.path.append('../../shared/')

from datasets import Dataset,DatasetDict
from knowledge_utils import *
from knowledge import *
from utils import *
from transformers import BartForConditionalGeneration
from transformers.trainer_utils import set_seed
from transformers.data.data_collator import DataCollatorForSeq2Seq

import os
import pickle
import pandas as pd
import numpy as np
import argparse

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

CONCEPTNET_EMBEDDING_FILE = 'data/numberbatch-en-19.08.txt'
EDGE_DATA_FILE = 'data/conceptnet-assertions-5.7.0.csv'

EDGE_DICT_FILE = 'data/edge_dictionary.pickle'
EMB_DICT_FILE = 'data/embedding_dictionary.pickle'

BART_HIDDEN_SIZE = 768
EMBEDDING_SIZE = 300
MAX_LENGTH = 1024

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    parser.add_argument('--seed', type=int, default=685, help='Pass in a seed value.')
    parser.add_argument('--batch_size', type=int, default=4, help='Pass in a batch size.')
    parser.add_argument('--k', type=int, default=5, help='Pass in a value for k.')
    parser.add_argument('--num_epochs', type=float, default=3.0, help='Pass in the number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Pass in the learning rate for training.')
    parser.add_argument('--sep', type=str, default=',', help='Separator for data file.')
    parser.add_argument('--model_type', type=str, choices=['input', 'attn'], required=True, help='Pass in a model type.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.trn.csv', help='Data File to load.')
    parser.add_argument('--dev_file', type=str, help='Dev File to load in case data split isn''t used.')
    
    parser.add_argument(
        '--knowledge_dropout',
        type=float,
        default=0.2,
        help='Dropout for Knowledge Layer.'
    )

    return parser.parse_args()

def check_args(args):
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')

def process_data(args, use_dev=False):
    print('loading and tokenizing data ...')
    if use_dev:
      df = pd.read_csv(args.dev_file, sep=args.sep, engine='python')
    else:
      df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
    
    df = clean_post(df)
    df = clean_target(df)

    df_post = df[['HITId', 'post']]
    df_post = df_post.drop_duplicates().reset_index(drop=True)
    
    print('processing edge file ...')
    if not os.path.exists(EDGE_DICT_FILE):
      edge_dict = process_edge_file(EDGE_DATA_FILE)
      pickle.dump(edge_dict, open(EDGE_DICT_FILE, 'wb'))
    else:
      edge_dict = pickle.load(open(EDGE_DICT_FILE, 'rb'))
    
    print('processing numberbatch embeddings ...')
    emb_dict = None
    if args.model_type == 'attn':
      if not os.path.exists(EMB_DICT_FILE):
        emb_dict = process_embedding_file(CONCEPTNET_EMBEDDING_FILE)
        pickle.dump(emb_dict, open(EMB_DICT_FILE, 'wb'))
      else:
        emb_dict = pickle.load(open(EMB_DICT_FILE, 'rb'))

    print('finding top k tuples ...')
    df_post = concat_top_k_tuples(df_post, edge_dict, emb_dict=emb_dict, emb_size=EMBEDDING_SIZE*2, k=args.k)
    df = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')
    
    dataset = Dataset.from_pandas(df)
    return dataset

if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    print(args)
    set_seed(args.seed)
    
    dataset = process_data(args)
    if args.dev_file is not None:
      dev_dataset = process_data(args, use_dev=True)
      datasets = DatasetDict({"train": dataset, "test": dev_dataset})
    else:
      datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    print('tokenizing data ...')
    tokenizer, tokenized = tokenize_textgen_df(datasets, SEQ2SEQ_MODEL_NAME, padding=False, max_length=MAX_LENGTH)

    print('initializing model ...')
    if args.model_type == 'input':
      model = BartForConditionalGeneration.from_pretrained(SEQ2SEQ_MODEL_NAME)
    elif args.model_type == 'attn':
      model = BartForConditionalGenerationKnowledgeModel.from_pretrained(
          SEQ2SEQ_MODEL_NAME,
          knowledge_dropout=args.knowledge_dropout,
      )
    model.train()
    if torch.cuda.is_available():
      model.cuda()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True, max_length=MAX_LENGTH)
    train(model, tokenized, data_collator=data_collator, batch_size=args.batch_size, num_epochs=args.num_epochs, learning_rate=args.lr)

