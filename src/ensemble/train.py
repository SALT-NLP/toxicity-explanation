import sys
sys.path.append('../../shared/')

from datasets import Dataset, DatasetDict
from ensemble_utils import *
from ensemble import *
from utils import *
from transformers import BartForConditionalGeneration,AutoModelForCausalLM,AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.data.data_collator import DataCollatorForSeq2Seq

import random
import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_type_choices = ['vanilla_ensemble', 'ordered', 'dist', 'count', 'count_group', 'unordered_count', 'vanilla_ensemble_view']

    parser.add_argument('--seed', type=int, default=685, help='Pass in a seed value.')
    parser.add_argument('--batch_size', type=int, default=4, help='Pass in a batch size.')
    parser.add_argument('--T', '-T', type=float, default=0.5, help='Temperature setting for MultiView model.')
    parser.add_argument('--sep', type=str, default=',', help='Separator for data file.')
    parser.add_argument('--num_views', type=int, default=6, choices=[3, 6, 9], help='The number of views used by the model.')
    parser.add_argument('--model_type', type=str, choices=model_type_choices, required=True, help='Pass in a model type.')
    parser.add_argument('--ensemble_files', nargs='+', required=False, help="The ensemble model files.")
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.trn.csv', help='Data File to load.')

    return parser.parse_args()

def print_args(args):
    print('arguments ...')
    print('seed: ', args.seed)
    print('T: ', args.T)
    print('num_views: ', args.num_views)
    print('batch_size: ', args.batch_size)
    print('model_type: ', args.model_type)
    print('ensemble_files: ', args.ensemble_files)

def check_args(args):
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')
    
    if len(args.ensemble_files) == 0:
      raise ValueError('You must pass pickle files using the ensemble_models flag.')

    if args.model_type == 'vanilla_ensemble_view' and args.batch_size % args.num_views != 0:
      raise ValueError("Batch Size must be a multiple of the number of views!")

if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    check_args(args)
    view_model = args.model_type == 'vanilla_ensemble_view'
    set_seed(args.seed)
    
    print('loading and tokenizing data ...')
    df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
    df = clean_post(df)
    df = clean_target(df)
    
    df_ensemble, view_tokens = read_pickle_files(args.ensemble_files, args.model_type, args.num_views)
    df = df.merge(df_ensemble, on='HITId', validate='many_to_one')
    df.drop(['post'], axis=1, inplace=True)
    # if args.num_views == 3:
    #   print(df.view_0.tolist()[75:76])
    #   print(df.view_1.tolist()[75:76])
    #   print(df.view_2.tolist()[75:76])
    # elif args.num_views == 6:
    #   print(df.view_0.tolist()[75:76])
    #   print(df.view_1.tolist()[75:76])
    #   print(df.view_2.tolist()[75:76])
    #   print(df.view_3.tolist()[75:76])
    #   print(df.view_4.tolist()[75:76])
    #   print(df.view_5.tolist()[75:76])
    # elif args.num_views == 9:
    #   print(df.view_0.tolist()[75:76])
    #   print(df.view_1.tolist()[75:76])
    #   print(df.view_2.tolist()[75:76])
    #   print(df.view_3.tolist()[75:76])
    #   print(df.view_4.tolist()[75:76])
    #   print(df.view_5.tolist()[75:76])
    #   print(df.view_6.tolist()[75:76])
    #   print(df.view_7.tolist()[75:76])
    #   print(df.view_8.tolist()[75:76])
    
    if view_model:
      df.drop(['prediction'], axis=1, inplace=True)
 
      df = df.melt(id_vars=['HITId','target'], ignore_index=False)
      df = df.rename_axis('idx').sort_values(by=['idx','HITId', 'variable']).reset_index()

      total_rows = df.idx.max() + 1
      idx_list = np.arange(total_rows)
      np.random.shuffle(idx_list)
 
      split_idx = int(total_rows * 0.8)
      train_df = df[df.idx.isin(idx_list[:split_idx])]
      test_df = df[df.idx.isin(idx_list[split_idx:])]
 
      train_df = train_df.drop(['variable', 'idx'], axis=1).rename(columns={'value':'post'}).reset_index(drop=True)
      test_df = test_df.drop(['variable', 'idx'], axis=1).rename(columns={'value':'post'}).reset_index(drop=True)
      datasets = DatasetDict({'train': Dataset.from_pandas(train_df), 'test': Dataset.from_pandas(test_df)})
    else:
      df.rename(columns={'prediction':'post'}, inplace=True)
      dataset = Dataset.from_pandas(df)
      datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    print('tokenizing data ...')
    tokenizer, tokenized = tokenize_bart_df(
      datasets,
      SEQ2SEQ_MODEL_NAME,
      padding=False,
      max_length=MAX_LENGTH,
      special_tokens=view_tokens,
    )

    view_token_map = tokenizer.get_added_vocab()
    view_token_ids = set(view_token_map.values())

    if view_model:
      tokenized = add_view_token_idx_col(tokenized, view_token_ids)
    
    print('initializing model ...')
    if view_model:
      model = BartForConditionalGenerationMultiViewModel.from_pretrained(
        SEQ2SEQ_MODEL_NAME,
        T=args.T,
        num_views=args.num_views,
        tokenizer=tokenizer,
      )
      model.resize_token_embeddings(len(tokenizer))
    else:
      model = BartForConditionalGeneration.from_pretrained(SEQ2SEQ_MODEL_NAME)
    
    model.train()
    if torch.cuda.is_available():
      model.cuda()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True, max_length=MAX_LENGTH)
    train(model, tokenized, data_collator=data_collator, batch_size=args.batch_size, view_model=view_model, num_views=args.num_views)
    
