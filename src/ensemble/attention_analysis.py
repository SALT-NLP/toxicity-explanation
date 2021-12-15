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
    parser.add_argument('--ensemble_files', nargs='+', required=False, help="The ensemble model files.")
    parser.add_argument('--data_file', default='../../data/SBIC.v2.dev.csv', help='Data file to load.')
    parser.add_argument('--hitid_file', help='Path to HITID file for generation.')
    parser.add_argument('--full_analysis', action='store_true', help='Full Dev Set Analysis.')

    return parser.parse_args()

def print_args(args):
    print('arguments ...')
    print('batch_size: ', args.batch_size)

def check_args(args):
    if len(args.ensemble_files) == 0:
      raise ValueError('You must pass pickle files using the ensemble_models flag.')
    
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')
    
    if args.batch_size % args.num_views != 0:
      raise ValueError("Batch Size must be a multiple of the number of views!")
    
    model_path = args.model
    model_name = get_file_name(model_path)
    
    if args.full_analysis:
      data_source = get_file_name(args.data_file)
    else:
      data_source = get_file_name(args.hitid_file)
    
    pickle_file = 'pred/' + model_name + '_' + data_source + '.pickle'
    print('Pickle File: ', pickle_file)

    return model_path, pickle_file

def print_attn(df, col='mixture'):
    col_one = df[(df[col] == 1)]
    col_zero = df[(df[col] == 0)]
    
    attn_one = np.array(col_one.view_attention.tolist())
    attn_zero = np.array(col_zero.view_attention.tolist())
    
    attn_one = np.concatenate([
        (attn_one[:,0] + attn_one[:,1])[:,None],
        (attn_one[:,2] + attn_one[:,3])[:,None],
        (attn_one[:,4] + attn_one[:,5])[:,None]],
        axis=1
    )
    attn_zero = np.concatenate([
        (attn_zero[:,0] + attn_zero[:,1])[:,None],
        (attn_zero[:,2] + attn_zero[:,3])[:,None],
        (attn_zero[:,4] + attn_zero[:,5])[:,None]],
        axis=1
    )
    
    attn_one_avg = np.average(attn_one, axis=0)
    attn_zero_avg = np.average(attn_zero, axis=0)
    attn_one_std = np.var(attn_one, axis=0)
    attn_zero_std = np.var(attn_zero, axis=0)

    print('Column: ', col)
    print('Col Value 1 (Mean): ', attn_one_avg)
    print('Col Value 1 (Var): ', attn_one_std)
    print('Col Value 0 (Mean): ', attn_zero_avg)
    print('Col Value 0 (Var): ', attn_zero_std)
    print()

def get_exact_match(df, col):
    df[col] = df.apply(lambda col: col['prediction'] in col['target'], axis=1)
    df.drop(columns=['post', 'target', 'prediction'], inplace=True)
    print(col)
    print(sum(df[col].tolist()))
    print(len(df[col].tolist()))
    print()
    return df

if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    model_path, pickle_file = check_args(args)
    model_type = 'vanilla_ensemble_view'
    view_model = True

    if not os.path.exists(pickle_file):
      print('loading and tokenizing data ...')
      df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
      df_hitid = pd.read_csv(args.hitid_file)
      
      df = clean_post(df)
      df = clean_target(df, train=False)

      df_ensemble, view_tokens = read_pickle_files(args.ensemble_files, model_type, args.num_views)
      df = df.merge(df_ensemble, on='HITId', validate='many_to_one')
      
      df.drop(['post'], axis=1, inplace=True)
      if view_model:
        df.drop(['prediction'], axis=1, inplace=True)
 
        df = df.melt(id_vars=['HITId','target'], ignore_index=False)
        df = df.rename_axis('idx').sort_values(by=['idx','HITId', 'variable']).reset_index()
        df = df.drop(['idx', 'variable'], axis=1).rename(columns={'value':'post'})
      else:
        df.rename(columns={'prediction':'post'}, inplace=True)
      df = df.merge(df_hitid, on='HITId', validate='many_to_one')
      
      dataset = Dataset.from_pandas(df)

      print('tokenizing data ...')
      tokenizer, tokenized = tokenize_bart_df(dataset, SEQ2SEQ_MODEL_NAME, padding=False, train=False, max_length=MAX_LENGTH, special_tokens=view_tokens)
      
      view_token_map = tokenizer.get_added_vocab()
      view_token_ids = set(view_token_map.values())
      tokenized = add_view_token_idx_col(tokenized, view_token_ids, train=False)

      print('initializing model ...')
      model = BartForConditionalGenerationMultiViewModel.from_pretrained(
        model_path,
        T=args.T,
        use_cuda=args.use_cuda,
        num_views = args.num_views
      )
      
      model.eval()
      if args.use_cuda and torch.cuda.is_available():
        model.cuda()
      
      input_cols = ['input_ids', 'view_token_idx', 'attention_mask']
      batch_iter = MinibatchIterator(tokenized, tokenizer, batch_size=args.batch_size, torch_cols=input_cols, use_cuda=args.use_cuda)
      enc_forward = model.encoder_view_forward
      
      results_cols_ext = []
      for column in df.columns:
        if column in {'expert','explicit','implicit','mixture'}:
          results_cols_ext.append(column)

      results_cols = ['HITId','post','target','view_attention']
      results_cols.extend(results_cols_ext)
      
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
    
    if args.full_analysis:
      results = pickle.load(open(pickle_file, 'rb'))
      expert_1_results = pickle.load(open(args.ensemble_files[0], 'rb'))
      expert_2_results = pickle.load(open(args.ensemble_files[0], 'rb'))
      explicit_1_results = pickle.load(open(args.ensemble_files[2], 'rb'))
      explicit_2_results = pickle.load(open(args.ensemble_files[3], 'rb'))
      implicit_1_results = pickle.load(open(args.ensemble_files[4], 'rb'))
      implicit_2_results = pickle.load(open(args.ensemble_files[5], 'rb'))
      
      expert_1_results = get_exact_match(expert_1_results, 'expert_1')
      expert_2_results = get_exact_match(expert_2_results, 'expert_2')
      explicit_1_results = get_exact_match(explicit_1_results, 'explicit_1')
      explicit_2_results = get_exact_match(explicit_2_results, 'explicit_2')
      implicit_1_results = get_exact_match(implicit_1_results, 'implicit_1')
      implicit_2_results = get_exact_match(implicit_2_results, 'implicit_2')
      
      results.drop(columns=['post','target','prediction'], inplace=True)
      results = results.merge(expert_1_results, on='HITId', validate='one_to_one')
      results = results.merge(expert_2_results, on='HITId', validate='one_to_one')
      results = results.merge(explicit_1_results, on='HITId', validate='one_to_one')
      results = results.merge(explicit_2_results, on='HITId', validate='one_to_one')
      results = results.merge(implicit_1_results, on='HITId', validate='one_to_one')
      results = results.merge(implicit_2_results, on='HITId', validate='one_to_one')

      results['expert'] = (results.expert_1) & (results.expert_2)
      results['explicit'] = (results.explicit_1) & (results.explicit_2)
      results['implicit'] = (results.implicit_1) & (results.implicit_2)
      
      print_attn(results, col='expert')
      print_attn(results, col='explicit')
      print_attn(results, col='implicit')
    else:
      results = pickle.load(open(pickle_file, 'rb'))
      #print_attn(results)
      #print_attn(results, col='expert')
      #print_attn(results, col='explicit')
      print_attn(results, col='implicit')

