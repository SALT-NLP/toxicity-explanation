import sys
import argparse
import pickle
import pandas as pd

from datasets import load_metric
from scipy.stats import wilcoxon, ttest_ind
from utils import *

gpt_pred_file_sbic_dev = '../baselines/pred/gpt_SBIC.v2.dev_pred.pickle'
gpt_actual_file_sbic_dev = '../baselines/pred/gpt_SBIC.v2.dev_actual.pickle'
gpt_pred_file_sbic_test = '../baselines/pred/gpt_SBIC.v2.tst_pred.pickle'
gpt_actual_file_sbic_test = '../baselines/pred/gpt_SBIC.v2.tst_actual.pickle'

gpt2_pred_file_sbic_dev = '../baselines/pred/gpt2_SBIC.v2.dev_pred.pickle'
gpt2_actual_file_sbic_dev = '../baselines/pred/gpt2_SBIC.v2.dev_actual.pickle'
gpt2_pred_file_sbic_test = '../baselines/pred/gpt2_SBIC.v2.tst_pred.pickle'
gpt2_actual_file_sbic_test = '../baselines/pred/gpt2_SBIC.v2.tst_actual.pickle'

bart_pred_file_sbic_dev = '../src/seq2seq/pred/bart_base_checkpoint-3epoch_SBIC.v2.dev.pickle'
bart_pred_file_sbic_test = '../src/seq2seq/pred/bart_base_checkpoint-3epoch_SBIC.v2.tst.pickle'
bart_pred_file_implicit_hate = '../src/seq2seq/pred/bart_base_checkpoint-3epoch_implicit_hate_v1_stg3_posts.tst.pickle'

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('--baseline', type=str, choices=['bart', 'gpt', 'gpt2'], required=True, help='Pass in a baseline model.')
  parser.add_argument('--dataset', type=str, choices=['sbic_dev', 'sbic_test', 'implicit_hate'], required=True, help='Pass in a dataset.')
  parser.add_argument('--cmp_file', type=str, required=True, help='Pass in pred file for a comparison model.')

  return parser.parse_args()

def normalize_gpt_file(pred_file, actual_file):
  base_pred_file = pickle.load(open(pred_file, 'rb'))
  base_actual_file = pickle.load(open(actual_file, 'rb'))
  
  base_pred_file.rename(columns={'targetStereotype':'prediction'}, inplace=True)
  base_pred_file = base_pred_file[['HITId', 'prediction']]
  
  base_actual_file.rename(columns={'targetStereotype':'target'}, inplace=True)
  base_actual_file = base_actual_file[['HITId', 'post', 'target']]
  
  df = base_actual_file.merge(base_pred_file, on='HITId', validate='one_to_one')
  return df

def run_bleu_statistics(base_hyp, cmp_hyp, ref):
  base_bleu = get_bleu_score(ref, base_hyp, return_all_scores=True)
  cmp_bleu = get_bleu_score(ref, cmp_hyp, return_all_scores=True)
  
  base_bleu = base_bleu[:,0].squeeze()
  cmp_bleu = cmp_bleu[:,0].squeeze()
  
  statistic, pval = wilcoxon(cmp_bleu, y=base_bleu, alternative='greater')
  #statistic, pval = ttest_ind(cmp_bleu, base_bleu, equal_var=False, alternative='greater')
  print('statistic (bleu): ', statistic)
  print('p-value (bleu): ', pval)

def run_rouge_statistics(base_hyp, cmp_hyp, ref):
  base_rouge = get_rouge_scores(ref, base_hyp, return_all_scores=True)
  cmp_rouge = get_rouge_scores(ref, cmp_hyp, return_all_scores=True)
  
  base_rouge = base_rouge[:,0,2].squeeze()
  cmp_rouge = cmp_rouge[:,0,2].squeeze()
  
  statistic, pval = wilcoxon(cmp_rouge, y=base_rouge, alternative='greater')
  #statistic, pval = ttest_ind(cmp_rouge, base_rouge, equal_var=False, alternative='greater')
  print('statistic (rouge): ', statistic)
  print('p-value (rouge): ', pval)

def run_bertscore_statistics(base_hyp, cmp_hyp, ref):
  metric = load_metric('bertscore')
  base_bert_scores = metric.compute(predictions=base_hyp, references=ref, lang='en')
  cmp_bert_scores = metric.compute(predictions=cmp_hyp, references=ref, lang='en')

  base_bert = get_bert_score(base_bert_scores, base_hyp, ref, return_all_scores=True)['f1']
  cmp_bert = get_bert_score(cmp_bert_scores, cmp_hyp, ref, return_all_scores=True)['f1']
  
  statistic, pval = wilcoxon(cmp_bert, y=base_bert, alternative='greater')
  #statistic, pval = ttest_ind(cmp_bert, base_bert, equal_var=False, alternative='greater')
  print('statistic (bertscore): ', statistic)
  print('p-value (bertscore): ', pval)

def run_significance_test(base_df, cmp_df):
  base_hyp = base_df['prediction'].tolist()
  cmp_hyp = cmp_df['prediction'].tolist()
  ref = base_df['target'].tolist()
  
  run_bleu_statistics(base_hyp, cmp_hyp, ref)
  run_rouge_statistics(base_hyp, cmp_hyp, ref)
  run_bertscore_statistics(base_hyp, cmp_hyp, ref)

if __name__ == "__main__":
  args = parse_args()
  
  cmp_df = pickle.load(open(args.cmp_file, 'rb'))
  if args.dataset == 'sbic_dev':
    if args.baseline == 'gpt':
      base_df = normalize_gpt_file(gpt_pred_file_sbic_dev, gpt_actual_file_sbic_dev)
    elif args.baseline == 'gpt2':
      base_df = normalize_gpt_file(gpt2_pred_file_sbic_dev, gpt2_actual_file_sbic_dev)
    else:
      base_df = pickle.load(open(bart_pred_file_sbic_dev, 'rb'))
  elif args.dataset == 'sbic_test':
    if args.baseline == 'gpt':
      base_df = normalize_gpt_file(gpt_pred_file_sbic_test, gpt_actual_file_sbic_test)
    elif args.baseline == 'gpt2':
      base_df = normalize_gpt_file(gpt2_pred_file_sbic_test, gpt2_actual_file_sbic_test)
    else:
      base_df = pickle.load(open(bart_pred_file_sbic_test, 'rb'))
  elif args.dataset == 'implicit_hate':
    if args.baseline == 'gpt' or args.baseline == 'gpt2':
      raise ValueError('Cannot run GPT tests on Implicit Hate Corpus.')
    else:
      base_df = pickle.load(open(bart_pred_file_implicit_hate, 'rb'))

  run_significance_test(base_df, cmp_df)

