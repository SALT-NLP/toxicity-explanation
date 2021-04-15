## This file is an alternative to the jupyter notebook in case training
## will be done from the command line instead.

import math
from datasets import Dataset
from torch import nn, torch
from transformers import Trainer, TrainingArguments
from seq2seq import BartForConditionalGenerationJoinModel
from seq2seq_utils import *

# Useful constants
CLASSIFIER_MODEL_NAME = 'bert-base-uncased'
CLASSIFIERS = ['./classification/model/whoTarget/checkpoint-1280/']
SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'
DATA_DIR = '../data/'
JOIN_DROPOUT = 0.0
SEED = 154
WARMUP_DIV = 9.793

def tokenize_df():
    df = pd.read_csv(DATA_DIR + 'SBIC.v2.trn.csv')
    df = clean_post(df)
    df = clean_target(df)
    df.to_csv('data/clean_train_df.csv')
    
    dataset = Dataset.from_pandas(df)
    datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    seq2seq_tok, classifier_tok, tokenized = get_tokenized_data(
        datasets,
        SEQ2SEQ_MODEL_NAME,
        CLASSIFIER_MODEL_NAME,
        labels=True,
    )
    
    return tokenized

def init_model():
    model = BartForConditionalGenerationJoinModel.from_pretrained(
                SEQ2SEQ_MODEL_NAME,
                join_dropout=JOIN_DROPOUT,
                classifiers=CLASSIFIERS,
            )
    
    if torch.cuda.is_available():
      model = model.cuda()
    
    model.train()
    return model

def get_step_variables(num_rows, num_epochs, batch_size):
    if num_epochs == 1:
      one_epoch_steps = math.ceil(num_rows / batch_size) // 2
      warmup_steps = (one_epoch_steps * num_epochs) // WARMUP_DIV
      save_steps = one_epoch_steps * 2
      eval_steps = (save_steps * 10.0) // 100
    else:
      one_epoch_steps = math.ceil(num_rows / batch_size)
      warmup_steps = (one_epoch_steps * num_epochs) // WARMUP_DIV
      save_steps = (one_epoch_steps * num_epochs) // 2
      eval_steps = (one_epoch_steps * num_epochs * 5.0) // 100

    return warmup_steps, save_steps, eval_steps
    

def train(model, tokenized):
    num_rows = tokenized['train'].num_rows
    num_epochs = 3.0

    learning_rate = 5e-6
    batch_size = 8 
    
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
    
    #trainer.train()

if __name__ == '__main__':
    print('preparing and tokenizing data ...')
    tokenized = tokenize_df()
    print('initializing model ...')
    model = init_model()
    train(model, tokenized)

