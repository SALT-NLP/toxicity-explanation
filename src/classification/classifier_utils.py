import pandas as pd
import numpy as np
import statistics as stats

from classifier_utils import *

def clean_post(df):
  df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
  df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
  df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
  df.post = df.post.str.strip()

def categorize_vars(df, classify_col):
  # Classify each annotator's rating
  print(df[classify_col].isna().any())
  df.dropna(axis=0, subset=[classify_col], inplace=True)
  df[classify_col] = np.where(df[classify_col] >= 0.5, 1, 0)
  
  # Sum classifications over annotator and choose the majority vote.
  df = df.groupby(['HITId', 'post'], as_index=False).agg({classify_col:['sum', 'count']})
  df.columns = ['HITId', 'post', 'sum', 'count']
  df[classify_col] = np.where(df['sum'] >= df['count'] / 2, 1, 0)

  return df[['post', classify_col]]

def prep_df_for_classification(df, to_file, classify_col):
  df = categorize_vars(df, classify_col)
  clean_post(df)
  df.to_csv(to_file, index=False)
  return df

def compute_statistics(sentences):
  token_lens = [len(sent) for sent in sentences]

  avg_tokens = sum(token_lens) / len(token_lens)
  max_tokens = max(token_lens)

  ntiles = np.quantile(token_lens, [i*0.01 for i in range(100)])
  stdev = stats.stdev(token_lens)

  print("Avg. Tokens: ", avg_tokens)
  print("Max Tokens: ", max_tokens)
  print("n-tiles: ", ntiles)
  print("Std Dev: ", stdev)



