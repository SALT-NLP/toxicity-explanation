import pandas as pd
import numpy as np
import statistics as stats

def clean_post(df):
  df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
  df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
  df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
  df.post = df.post.str.strip()
  return df

def categorize_vars(df, classify_col, colY, colN):
  # Classify each annotator's rating
  #print(df[classify_col].isna().any())
  df.dropna(axis=0, subset=[classify_col], inplace=True)
  df[classify_col] = np.where(df[classify_col] >= 0.5, 1, 0)
  
  # Sum classifications over annotator and choose the majority vote.
  df = df.groupby(['HITId', 'post'], as_index=False).agg({classify_col:['sum', 'count']})
  df.columns = ['HITId', 'post', 'sum', 'count']
  df[classify_col] = np.where(df['sum'] >= df['count'] / 2, colY, colN)

  return df[['HITId', 'post', classify_col]]

def prep_df_for_classification(df, classify_col, colY, colN, hitids=None):
  df = categorize_vars(df, classify_col, colY, colN)
  df = clean_post(df)
  
  if hitids is not None:
    df = df[df.HITId.isin(hitids)].reset_index()
  
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



