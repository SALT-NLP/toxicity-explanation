"""
This file contains data cleaning functions for the SBIC corpus

See the SBIC Corpus here:
  https://homes.cs.washington.edu/~msap/social-bias-frames/DATASTATEMENT.html
"""
import numpy as np

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
  df.targetStereotype = df.targetStereotype.str.replace(r'\bfolks?\b', 'people', regex=True)

  minority_row = ~(df.targetMinority.str.contains(',', na=False))
  stereotype_row = df.targetStereotype.str.contains(r'^\bare(n\'t)?\b', na=False, regex=True)
  row_repl = minority_row & stereotype_row
  
  df_incomplete = df.loc[row_repl]
  repl_str = df_incomplete.targetMinority + ' ' + df_incomplete.targetStereotype
  df.loc[row_repl, "targetStereotype"] = repl_str
  

def clean_target_minority(df):
  df.targetMinority = df.targetMinority.replace(np.nan, '', regex=True)
  df.targetMinority = df.targetMinority.str.lower()
  df.targetMinority = df.targetMinority.str.replace(r'[.?!]$', '', regex=True)
  df.targetMinority = df.targetMinority.str.replace(r'\bfolks?\b', 'people', regex=True)
  df.targetMinority = df.targetMinority.str.replace(r'^\'|^"|"$|\'$','', regex=True)

def clean_post(df):
  df.post = df.post.str.replace('RT ', ' ', regex=False)
  df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
  df.post = df.post.str.replace('(&#[0-9]+|&[a-zA-Z0-9]+);', ' ', regex=True)
  df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
  df.post = df.post.str.strip()

def create_text_column(df):
  df.sexYN = np.where(df.sexYN >= 0.5, LEWDY, LEWDN)
  df.offensiveYN = np.where(df.offensiveYN >= 0.5, OFFY, OFFN)
  df.intentYN = np.where(df.intentYN >= 0.5, INTY, INTN)
  df.whoTarget = np.where(df.whoTarget >= 0.5, GRPY, GRPN)
  df.speakerMinorityYN = np.where(df.speakerMinorityYN >= 0.5, INGY, INGN)

  df['text'] = BOS + df.post + SEP + df.sexYN + ' ' + df.offensiveYN + ' ' + \
                  df.intentYN + ' ' + df.whoTarget + SEP + df.targetMinority + \
                  SEP + df.targetStereotype + SEP + df.speakerMinorityYN + EOS

