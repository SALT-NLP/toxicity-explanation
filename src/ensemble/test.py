import sys
sys.path.append('../../shared/')

from datasets import Dataset
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
    model_type_choices = ['vanilla_ensemble', 'ordered', 'dist', 'count', 'count_group', 'unordered_count', 'vanilla_ensemble_gate', 'vanilla_ensemble_view']
    
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='The path for the checkpoint folder',
    )
    
    parser.add_argument('--batch_size', type=int, default=4, help='Pass in a batch size.')
    parser.add_argument('--T', '-T', type=float, default=0.5, help='Temperature setting for MultiView model.')
    parser.add_argument('--num_views', type=int, default=6, choices=[3, 6, 9], help='The number of views used by the model.')
    parser.add_argument('--sep', type=str, default=',', help='Separator for data file.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    parser.add_argument('--generate_scores', action='store_true', help='If True, will generate scores')
    parser.add_argument('--model_type', type=str, choices=model_type_choices, required=True, help='Pass in a model type.')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--ensemble_files', nargs='+', required=False, help="The ensemble model files.")
    parser.add_argument('--data_file', default='../../data/SBIC.v2.dev.csv', help='Data file to load.')
    parser.add_argument('--hitid_file', help='Path to HITID file for generation. (CURRENTLY NOT IN USE)')

    return parser.parse_args()

def print_args(args):
    print('arguments ...')
    print('batch_size: ', args.batch_size)
    print('model_type: ', args.model_type)

def check_args(args):
    if len(args.ensemble_files) == 0:
      raise ValueError('You must pass pickle files using the ensemble_models flag.')
    
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')
    
    if args.model_type == 'vanilla_ensemble_view' and args.batch_size % args.num_views != 0:
      raise ValueError("Batch Size must be a multiple of the number of views!")
    
    model_path = args.model
    model_name = get_file_name(model_path)
    
    data_source = get_file_name(args.data_file)
    pickle_file = 'pred/' + model_name + '_' + data_source + '.pickle'
    results_file = 'results/' + model_name + '_' + data_source + '.csv'

    return model_path, pickle_file, results_file

if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    model_path, pickle_file, results_file = check_args(args)
    view_model = args.model_type == 'vanilla_ensemble_view'

    if not os.path.exists(pickle_file):
      print('loading and tokenizing data ...')
      df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
      df = clean_post(df)
      df = clean_target(df, train=False)

      df_ensemble, view_tokens = read_pickle_files(args.ensemble_files, args.model_type, args.num_views)
      df = df.merge(df_ensemble, on='HITId', validate='many_to_one')
      
      df.drop(['post'], axis=1, inplace=True)
      if view_model:
        df.drop(['prediction'], axis=1, inplace=True)
 
        df = df.melt(id_vars=['HITId','target'], ignore_index=False)
        df = df.rename_axis('idx').sort_values(by=['idx','HITId', 'variable']).reset_index()
        df = df.drop(['idx', 'variable'], axis=1).rename(columns={'value':'post'})
      else:
        df.rename(columns={'prediction':'post'}, inplace=True)
      
      dataset = Dataset.from_pandas(df)

      print('tokenizing data ...')
      tokenizer, tokenized = tokenize_bart_df(dataset, SEQ2SEQ_MODEL_NAME, padding=False, train=False, max_length=MAX_LENGTH, special_tokens=view_tokens)
      
      view_token_map = tokenizer.get_added_vocab()
      view_token_ids = set(view_token_map.values())
      
      if view_model:
        tokenized = add_view_token_idx_col(tokenized, view_token_ids, train=False)
      
      print('initializing model ...')
      if view_model:
        model = BartForConditionalGenerationMultiViewModel.from_pretrained(
          model_path,
          T=args.T,
          use_cuda=args.use_cuda,
          num_views = args.num_views
        )
      else:
        model = BartForConditionalGeneration.from_pretrained(model_path)
      
      model.eval()
      if args.use_cuda and torch.cuda.is_available():
        model.cuda()
      
      input_cols = ['input_ids', 'view_token_idx', 'attention_mask'] if view_model else ['input_ids', 'attention_mask']
      batch_iter = MinibatchIterator(tokenized, tokenizer, batch_size=args.batch_size, torch_cols=input_cols, use_cuda=args.use_cuda)
      enc_forward = model.encoder_view_forward if view_model else model.get_encoder().forward
      
      results_cols = ['HITId','post','target', 'view_attention'] if view_model else ['HITId','post','target']
      generate_stereotypes(
          batch_iter,
          tokenizer,
          model,
          enc_forward,
          results_cols=results_cols,
          pickle_file=pickle_file,
          num_views=args.num_views,
          view_model=view_model
      )
    
    if args.generate_scores or args.save_results_to_csv:
      print("generating base model scores ...")
      generate_scores(
        pickle_file,
        save_results_to_csv=args.save_results_to_csv,
        num_results=args.num_results,
        save_file=results_file,
        view_model=view_model
      )

