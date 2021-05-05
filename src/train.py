## This file is an alternative to the jupyter notebook in case training
## will be done from the command line instead.

import math
from torch import nn, torch
from transformers import Trainer, TrainingArguments
from transformers import BartForConditionalGeneration
from seq2seq import BartForConditionalGenerationJoinModel
from tqdm import tqdm
from seq2seq_utils import *

# Useful constants
CLASSIFIER_MODEL_NAME = 'bert-base-uncased'
CLASSIFIERS = ['./classification/model/offensiveYN/checkpoint-1798/']
              #['./classification/model/whoTarget/checkpoint-1280/']
              #['./classification/model/sexYN/checkpoint-898/']
SEQ2SEQ_TOK_NAME = 'facebook/bart-base'
#SEQ2SEQ_MODEL_NAME = './model/bart_base_checkpoint-17970'
SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

DATA_DIR = '../data/'
FROM_DATA = DATA_DIR + 'SBIC.v2.trn.csv'
TO_DATA = 'data/clean_train_df.csv'

JOIN_DROPOUT = 0.2
SEED = 154
WARMUP_DIV = 9.793

def init_model(model_name, join=True, train=True, use_cuda=True):
    if join:
      model = BartForConditionalGenerationJoinModel.from_pretrained(
                  model_name,
                  join_dropout=JOIN_DROPOUT,
                  classifiers=CLASSIFIERS,
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

def get_step_variables(num_rows, num_epochs, batch_size):
    if num_epochs == 1:
      warmup_steps = math.ceil(num_rows / batch_size) // 2
      save_steps = warmup_steps * 2
      eval_steps = (save_steps * 5.0) // 100
    else:
      warmup_steps = math.ceil(num_rows / batch_size)
      save_steps = (warmup_steps * num_epochs) // 2
      eval_steps = (warmup_steps * num_epochs * 5.0) // 100
    return warmup_steps, save_steps, eval_steps
    
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

if __name__ == '__main__':
    print('preparing and tokenizing data ...')
    _, _, tokenized = tokenize_df(
        FROM_DATA,
        TO_DATA,
        SEQ2SEQ_TOK_NAME,
        CLASSIFIER_MODEL_NAME
    )
    
    print('initializing model ...')
    model = init_model(SEQ2SEQ_MODEL_NAME)
    train(model, tokenized)
