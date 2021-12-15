import sys
sys.path.append('../../../shared/')

import warnings
import torch
import pandas as pd
import numpy as np
import argparse
import os
import math
import pickle

from classifier_utils import *
from utils import *
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

DATA_DIR = '../../../data/'
BASE_MODEL = 'bert-base-uncased'

COL_DICT = {
  'intentYN': (INTY, INTN),
  'offensiveYN': (OFFY, OFFN),
  'sexYN': (LEWDY, LEWDN),
  'whoTarget': (GRPY, GRPN)
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='The path for the checkpoint folder',
    )

    parser.add_argument('--predict', action='store_true', help='Whether or not to run predictions. If False, will look for a prediction file.')
    parser.add_argument('--hitid_file', help='Path to HITID file for generation.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.dev.csv', help='Data File to load. Default: \'../../data/SBIC.v2.dev.csv\'')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    parser.add_argument('--generate_scores', action='store_true', help='If True, will generate scores')

    return parser.parse_args()

def check_args(args):
    use_hitid = args.hitid_file is not None
    model_path = args.model
    classify_col = os.path.basename(os.path.dirname(os.path.normpath(model_path)))
    
    data_source = get_file_name(args.data_file)
    pickle_file = 'pred/' + classify_col + '_' + data_source
    pickle_file = pickle_file + '.pickle' if args.hitid_file is None else pickle_file + '_hitid.pickle'

    if args.predict:
      if os.path.isfile(pickle_file):
        warnings.warn(pickle_file + ' exists and will be overwritten', RuntimeWarning)
    else:
      if not os.path.isfile(pickle_file):
        raise ValueError(pickle_file + ' does not exist. Run again with predict flag set to True')
      if use_hitid:
        warnings.warn(hitid_file + ' may not be used to filter data, since predict flag was not passed.')
    
    return pickle_file, classify_col

if __name__ == '__main__':
    args = parse_args()
    pickle_file, classify_col = check_args(args)
    
    if args.predict:
      hitids = None
      if args.hitid_file is not None:
        hitids = get_hitids(args.hitid_file)
      
      colY = COL_DICT[classify_col][0]
      colN = COL_DICT[classify_col][1]

      df = pd.read_csv(DATA_DIR + args.data_file)
      df = prep_df_for_classification(df, classify_col, colY, colN, hitids)

      tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
      model = BertForSequenceClassification.from_pretrained(args.model)
      model.eval()

      num_examples = df.shape[0]
      outputs = np.empty((num_examples, 2))
      
      for i in tqdm(range(num_examples)):
        inputs = tokenizer(df.post[i], return_tensors='pt')
        output = model(**inputs)
        outputs[i] = output['logits'].detach().cpu().numpy()
      
      outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:,None]
      outputs = np.argmax(outputs, axis=1)
      
      df.loc[:,'pred'] = outputs
      df.pred = np.where(df.pred >= 0.5, colY, colN)
      pickle.dump(df, open(pickle_file, 'wb'))

