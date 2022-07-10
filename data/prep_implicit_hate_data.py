import pandas as pd
import random
import pickle
import argparse
import os

from sklearn.model_selection import train_test_split

STAGE3_DATA = 'implicit-hate-corpus/implicit_hate_v1_stg3_posts.tsv'

def get_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', type=int, default=475, help='Random Seed for Data Split.')
  parser.add_argument('--start', type=int, default=100, help='Start index for HITId column creation')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_args()
  
  df = pd.read_csv(STAGE3_DATA, delimiter='\t')
  df = df.dropna(axis=0)
  df.rename(columns={'implied_statement':'targetStereotype', 'target':'targetMinority'}, inplace=True)
  
  post_df = df[['post']].copy()
  post_df['HITId'] = list(range(100, 100 + len(post_df)))
  post_df = post_df.drop_duplicates(subset = 'post')
  #df = post_df.merge(df)
  
  post_df_train, post_df_test = train_test_split(post_df, test_size=0.25, random_state=args.seed)
  post_df_test, post_df_dev = train_test_split(post_df_test, test_size=0.5, random_state=args.seed)
  train, dev, test = post_df_train.merge(df), post_df_dev.merge(df), post_df_test.merge(df)
  print('Train: \n', train)
  print('Test: \n', test)
  print('Dev: \n', dev)

  train.to_csv('implicit_hate_v1_stg3_posts.trn.tsv', sep='\t')
  test.to_csv('implicit_hate_v1_stg3_posts.tst.tsv', sep='\t')
  dev.to_csv('implicit_hate_v1_stg3_posts.dev.tsv', sep='\t')

