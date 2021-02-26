"""
This file contains data cleaning functions for the SBIC corpus

See the SBIC Corpus here:
  https://homes.cs.washington.edu/~msap/social-bias-frames/DATASTATEMENT.html
"""
import numpy as np
import pandas as pd

# Load tokenizers
from transformers import AutoTokenizer

# String Separator
SEP = '<sep>'
BOS = '<str>'
EOS = '<end>'
PAD = '<pad>'

# Categorical Special Tokens
LEWDY = '<lewdY>'
LEWDN = '<lewdN>'
OFFY = '<offY>'
OFFN = '<offN>'
INTY = '<intY>'
INTN = '<intN>'
GRPY = '<grpY>'
GRPN = '<grpN>'
INGY = '<ingY>'
INGN = '<ingN>'

def print_head(df):
  print("Data Head")
  print(df.head())
  print("\n")

def print_uniques(column):
  print("Unique Values")
  print(column.nunique())
  print("Total Values")
  print(column.count())
  print("\n")

def print_column_freq(column):
  print("Target Stereotype Freq.")
  print(column.value_counts())
  print("\n")

def clean_target_stereotype(df):
  df.targetStereotype = df.targetStereotype.replace(np.nan, '', regex=True)
  df.targetStereotype = df.targetStereotype.str.replace(r'[.?!]$', '', regex=True)

def clean_target_minority(df):
  df.targetMinority = df.targetMinority.replace(np.nan, '', regex=True)
  df.targetMinority = df.targetMinority.str.replace(r'[.?!]$', '', regex=True)

def clean_post(df):
  df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
  df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
  df.post = df.post.str.replace('(&#[0-9]+|&[a-zA-Z0-9]+);', ' ', regex=True)
  df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
  df.post = df.post.str.strip()

def categorize_var(df):
  df.sexYN = np.where(df.sexYN >= 0.5, LEWDY, LEWDN)
  df.offensiveYN = np.where(df.offensiveYN >= 0.5, OFFY, OFFN)
  df.intentYN = np.where(df.intentYN >= 0.5, INTY, INTN)
  df.whoTarget = np.where(df.whoTarget >= 0.5, GRPY, GRPN)
  df.speakerMinorityYN = np.where(df.speakerMinorityYN >= 0.5, INGY, INGN)

def create_text_column(df):
  categorize_var(df)
  df['text'] = BOS + df.post + SEP + df.sexYN + ' ' + df.offensiveYN + ' ' + \
                  df.intentYN + ' ' + df.whoTarget + SEP + df.targetMinority + \
                  SEP + df.targetStereotype + SEP + df.speakerMinorityYN + EOS

def clean_df(from_file, to_file):
  df = pd.read_csv(from_file)
  
  clean_target_minority(df)
  clean_target_stereotype(df)
  clean_post(df)
  create_text_column(df)
  
  df[['text']].sample(frac=1).to_csv(to_file, index=False)
  df[['text']].to_csv(to_file, index=False)

def setup_tokenizer(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
  categorical_tokens = [LEWDY,LEWDN,OFFY,OFFN,INTY,INTN,GRPY,GRPN,INGY,INGN]
  special_tokens = {'sep_token': SEP, \
                    'bos_token': BOS, \
                    'eos_token': EOS, \
                    'pad_token': PAD, \
                    'additional_special_tokens': categorical_tokens}
  tokenizer.add_special_tokens(special_tokens)
  return tokenizer
