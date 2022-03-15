## This file is an alternative to the jupyter notebook in case training
## will be done from the command line instead.
import sys
sys.path.append('../../shared/')

import math
import argparse
from torch import nn, torch
from transformers import Trainer, TrainingArguments
from transformers import BartForConditionalGeneration
from transformers.trainer_utils import set_seed
from datasets import DatasetDict
from seq2seq_utils import *
from utils import *

# Useful constants
CLASSIFIER_TOK_NAME = 'bert-base-uncased'
#CLASSIFIERS = [
#                './classification/model/offensiveYN/checkpoint-1798/',
#                './classification/model/whoTarget/checkpoint-1280/',
#                './classification/model/sexYN/checkpoint-898/',
#                './classification/model/intentYN/checkpoint-898/',
#              ]
SEQ2SEQ_TOK_NAME = 'facebook/bart-base'
SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--join', action='store_true', help='Trains BART with Join Embedding.')
    parser.add_argument('--sep', type=str, default=',', help='Pass in a separator for the data file.')
    parser.add_argument('--batch_size', type=int, default=4, help='Pass in a batch size.')
    parser.add_argument('--seed', type=int, default=193, help='Pass in a seed value.')
    parser.add_argument(
        '--join_dropout',
        type=float,
        default=0.2,
        help='Dropout for Join Embedding Params.'
    )
    parser.add_argument(
        '-c',
        '--classifiers',
        nargs='+',
        required=False,
        help='The path for the classifier (you can pass multiple)',
    )
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.trn.csv', help='Data File to load.')
    parser.add_argument('--dev_file', type=str, help='Dev File to load in case data split isn''t used.')
    parser.add_argument('--num_epochs', type=float, default=5.0, help='Pass in the number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Pass in the learning rate for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.5, help='Pass in the warmup ratio (only applies when training 1 epoch).')
    return parser.parse_args()

def check_args(args):
    if not(os.path.isfile(args.data_file)):
      raise ValueError('Must pass in an existing data file for training.')
    
    if args.join_dropout < 0.0 or args.join_dropout > 1.0:
      raise ValueError('Join Dropout must be between 0.0 and 1.0 (inclusive)')
    
    if args.join and args.classifiers is None:
      raise ValueError('You have selected a join model, but have not provided any classifiers as args.')

def process_df(args, data_file):
    print("cleaning csv ...")
    dataset = read_and_clean_csv(data_file, sep=args.sep)

    num_classifiers = 0
    num_classification_heads = 0

    if args.join:
      print("tokenizing and classifying data ...")
      attentions = get_classifier_attention(dataset, CLASSIFIER_TOK_NAME, args.classifiers)
      
      print("mapping classifier attention to dataset ...")
      dataset = map_column_to_dataset(dataset, attentions, 'classifier_attention')
      
      num_classifiers = attentions.shape[1]
      num_classification_heads = attentions.shape[2]

    print('tokenizing data for bart ...')
    _, dataset = tokenize_textgen_df(
        dataset,
        SEQ2SEQ_TOK_NAME,
    )
    return dataset, num_classifiers, num_classification_heads

if __name__ == '__main__':
    args = parse_args()
    check_args(args)

    set_seed(args.seed)
    print("Seed: ", args.seed)
    
    dataset, num_classifiers, num_classification_heads = process_df(args, args.data_file)
    if args.dev_file is None:
      datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    else:
      dev_dataset, _, _ = process_df(args, args.dev_file)
      datasets = DatasetDict({'train': dataset, 'test': dev_dataset})

    print('initializing model ...')
    model = init_model(
        SEQ2SEQ_MODEL_NAME,
        join=args.join,
        join_dropout=args.join_dropout,
        num_classifiers=num_classifiers,
        num_classification_heads=num_classification_heads,
    )
    train(
      model,
      datasets,
      batch_size=args.batch_size,
      num_epochs=args.num_epochs,
      learning_rate=args.lr,
      warmup_ratio=args.warmup_ratio,
      eval_percent=5.0
    )
