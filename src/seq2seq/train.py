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
from seq2seq import BartForConditionalGenerationJoinModel
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

FROM_DATA_FILE = '../../data/SBIC.v2.trn.csv'
TO_DATA_FILE = 'data/clean_train_df.csv'

JOIN_DROPOUT = 0.2
SEED = 193
 
def train(model, tokenized):
    num_rows = tokenized['train'].num_rows
    num_epochs = 3.0

    learning_rate = 5e-6
    batch_size = 4
    
    warmup_steps, save_steps, eval_steps = get_step_variables(
        num_rows, num_epochs, batch_size
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
        num_train_epochs = num_epochs,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )
    
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--join', action='store_true', help='Trains BART with Join Embedding.')
    parser.add_argument('--seed', type=int, default=SEED, help='Pass in a seed value. If nothing is passed a default of 193 is used.')
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
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.join_dropout < 0.0 or args.join_dropout > 1.0:
      raise ValueError('Join Dropout must be between 0.0 and 1.0 (inclusive)')
    
    if args.join and args.classifiers is None:
      raise ValueError('You have selected a join model, but have not provided any classifiers as args.')
    
    set_seed(args.seed)
    print("Seed: ", args.seed)
    
    print("cleaning csv ...")
    dataset = read_and_clean_csv(FROM_DATA_FILE, TO_DATA_FILE)

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
    _, dataset = tokenize_bart_df(
        dataset,
        SEQ2SEQ_TOK_NAME,
    )
    datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    print('initializing model ...')
    model = init_model(
        SEQ2SEQ_MODEL_NAME,
        join=args.join,
        join_dropout=JOIN_DROPOUT,
        num_classifiers=num_classifiers,
        num_classification_heads=num_classification_heads,
    )
    train(model, datasets)
