import sys
sys.path.append('../../shared/')

from tqdm import tqdm
from transformers import BartForConditionalGeneration,AutoModelForCausalLM,AutoTokenizer
from datasets import Dataset
from knowledge import *
from utils import *
from datasets import load_metric
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize,tokenize
from nltk.stem import WordNetLemmatizer
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import string
import torch
import pickle
import os
import random
import numpy as np
import pandas as pd

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

CONCEPTNET_EMBEDDING_FILE = 'data/numberbatch-en-19.08.txt'
EDGE_DATA_FILE = 'data/conceptnet-assertions-5.7.0.csv'

EDGE_DICT_FILE = 'data/edge_dictionary.pickle'
EMB_DICT_FILE = 'data/embedding_dictionary.pickle'

BART_HIDDEN_SIZE = 768
EMBEDDING_SIZE = 300
MAX_LENGTH = 1024

ADJECTIVE_TAGS = {'JJ','JJR','JJS'}
NOUN_TAGS = {'NN','NNS','NNP','NNPS'}
VERB_TAGS = {'VB','VBG','VBN','VBZ'}
STOPWORDS = stopwords.words('english')

SEP_TOKEN = '</s>'

REL_DICT = {
    'RelatedTo': ' is related to ',
    'FormOf': ' is a form of ',
    'IsA': ' is a ',
    'PartOf': ' is a part of ',
    'HasA': ' has a ',
    'UsedFor': ' is used for ',
    'CapableOf': ' is capable of ',
    'AtLocation': ' is at ',
    'Causes': ' causes ',
    'HasSubevent': ' happens with ',
    'HasFirstSubevent': ' begins with ',
    'HasLastSubevent': ' concludes with ',
    'HasPrerequisite': ' requires ',
    'HasProperty': ' has property ',
    'MotivatedByGoal': ' is motivated by the goal ',
    'ObstructedBy': ' is obstructed by ',
    'Desires': ' desires ',
    'CreatedBy': ' is created by ',
    'Synonym': ' is similar in meaning to ',
    'Antonym': ' is opposite in meaning to ',
    'DistinctFrom': ' is distinct from ',
    'DerivedFrom': ' is derived from ',
    'SymbolOf': ' symbolically represents ',
    'DefinedAs': ' is said to be ',
    'MannerOf': ' is a type of ',
    'LocatedNear': ' is typically near ',
    'HasContext': ' is used in the context of ',
    'SimilarTo': ' is similar to ',
    'EtymologicallyRelatedTo': ' has a similar origin to ',
    'EtymologicallyDerivedFrom': ' is derived in meaning from ',
    'CausesDesire': ' causes someone to desire ',
    'MadeOf': ' is made of ',
    'ReceivesAction': ' can receive ',
    'ExternalURL': ' can be understood further at ',
    'InstanceOf': ' is an example of ',
    'Entails': ' entails that ',
    'NotDesires': ' does not desire ',
    'NotUsedFor': ' is not used for ',
    'NotCapableOf': ' is not capable of ',
    'NotHasProperty': ' does not have property ',
}

PROMPTS = [
  'were known for', 'were described as', 'were regarded as',
  'were thought of as', 'were well-known for', 'were seen as a',
  'have a reputation for', 'would frequently', 'worked as',
  'have a job as', 'have a part-time job as', 'earned money by',
  'started working as', 'have various hobbies such as', 'would regularly engage in',
  'frequently talked about', 'behaved as though', 'liked to',
]


# Explicit Knowledge Utils

class LemmaTokenizer:
    def __init__(self):
      self.ignore = {',', '.', ';', ':', '"', '``', "''", '`'}.union(STOPWORDS)
      self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
      return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore]

def add_queries(query_lists, query_tokens, post, lemmatizer):
    post_tok = word_tokenize(post)
    pos_tags = nltk.pos_tag(post_tok)
    query_lists.append([])
    
    for tag_tuple in pos_tags:
      tag = tag_tuple[1]
      word = tag_tuple[0]
      
      pos = ''
      if tag in ADJECTIVE_TAGS:
        pos = 'a'
      elif tag in NOUN_TAGS:
        pos = 'n'
      elif tag in VERB_TAGS:
        pos = 'v'

      if pos != '':
        word = lemmatizer.lemmatize(word, pos)
        if tag != 'NNP' and tag != 'NNPS':
          word = word.lower()
        
        if word not in STOPWORDS:
          query_lists[-1].append(word)
          query_tokens.add(word)
    
def process_edge_file(edge_data_file):
    edge_dict = defaultdict(list)
    with open(edge_data_file, 'r') as edge_data:
      for line in tqdm(edge_data):
        line = line.strip().split('\t')
        start = line[2].split('/')
        end = line[3].split('/')
        
        if start[2] == 'en' and end[2] == 'en':
          node_info = literal_eval(line[4])
          relation = line[1].split('/')
          
          if relation[2] == 'dbpedia':
            continue
          
          edge_dict[start[3]].append((relation[2], node_info['weight'], end[3]))
    return edge_dict

def process_embedding_file(emb_data_file):
    emb_dict = defaultdict(list)
    with open(emb_data_file, 'r') as emb_file:
      txt = emb_file.readline()
      for line in tqdm(emb_file):
        line = line.split(' ')
        line[-1] = line[-1].strip()
        
        key = line[0]
        weights = list(map(float, line[1:]))
        emb_dict[key] = weights
     
    return emb_dict

def get_idf(posts, vocabulary):
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), vocabulary=vocabulary)
    vectorizer.fit_transform(posts)
    return dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def collect_edges(query_tokens, idf, edge_dict, emb_dict=None):
    query_dict = defaultdict(list)

    for query in tqdm(list(query_tokens)):
      idf_score = idf[query]
      
      for edge_info in edge_dict[query]:
        score = idf_score * edge_info[1]
        
        if emb_dict is not None:
          if query not in emb_dict or edge_info[2] not in emb_dict:
            continue
        
        query_dict[query].append((score, query, edge_info[0], edge_info[2]))
    
    return query_dict

def collect_ordered_k_tuples(query_lists, query_dict, k=5):
    ordered_k_tuples = []
    for query_list in tqdm(query_lists):
      ordered_k_tuple = set()
      
      for query in query_list:
        ordered_k_tuple = ordered_k_tuple.union(query_dict[query])
      
      ordered_k_tuple = list(ordered_k_tuple)
      ordered_k_tuple.sort(reverse=True)
      
      ordered_k_tuples.append(ordered_k_tuple[:k])
    return ordered_k_tuples

def add_ordered_k_tuples(posts, ordered_k_tuples, sep_token=SEP_TOKEN):
    for i,post in enumerate(tqdm(posts)):
      ordered_k_tuple = ordered_k_tuples[i]
      edge_string = ''
      for edge in ordered_k_tuple:
        start = edge[1]
        relation = REL_DICT[edge[2]]
        end = edge[3].replace('_', ' ')

        edge_string += sep_token + edge[1] + relation + end
      posts[i] += edge_string
    return posts

def add_ordered_k_tuples_embeds(posts, ordered_k_tuples, emb_dict, emb_size=600, k=5):
    embeddings = []
    for i,post in enumerate(tqdm(posts)):
      ordered_k_tuple = ordered_k_tuples[i]
      embeddings.append([])
      
      for edge in ordered_k_tuple:
        embed = []
        embed.extend(emb_dict[edge[1]])
        embed.extend(emb_dict[edge[3]])
        embeddings[-1].append(embed)
      
      for _ in range(k - len(embeddings[-1])):
        embed = emb_size * [0.0]
        embeddings[-1].append(embed)
      
    return posts, embeddings

def concat_top_k_tuples(
    df_post,
    edge_dict,
    sep_token=SEP_TOKEN,
    emb_dict=None,
    emb_size=600,
    k=5
):
    posts = df_post['post'].tolist()
    lemmatizer = WordNetLemmatizer()
    query_lists = []
    query_tokens = set()

    for post in tqdm(posts):
      add_queries(query_lists, query_tokens, post, lemmatizer)

    idf = get_idf(posts, query_tokens)
    query_dict = collect_edges(query_tokens, idf, edge_dict, emb_dict)
    ordered_k_tuples = collect_ordered_k_tuples(query_lists, query_dict, k=k)

    if emb_dict is None:
      posts = add_ordered_k_tuples(posts, ordered_k_tuples, sep_token)
    else:
      posts, embeddings = add_ordered_k_tuples_embeds(posts, ordered_k_tuples, emb_dict, emb_size=emb_size, k=k)
      df_post['knowledge_embeds'] = embeddings
    
    df_post['post'] = posts
    return df_post

# Implicit Knowledge Utils

def run_target_minority_model(df, tm_pred_pickle_file, args):
    datasets = Dataset.from_pandas(df)
    
    tokenizer, tokenized = tokenize_bart_df(datasets, SEQ2SEQ_MODEL_NAME, padding=False, train=False, max_length=MAX_LENGTH)
    batch_iter = MinibatchIterator(tokenized, tokenizer, batch_size=8)
     
    model = BartForConditionalGeneration.from_pretrained(args.target_minority_model)
    model.eval()
    model.cuda()
    
    results = generate_stereotypes(batch_iter, tokenizer, model, model.get_encoder().forward, pickle_file=tm_pred_pickle_file)
    return results

def run_implicit_generation(tm_pred, args):
    target_minorities = tm_pred.loc[tm_pred['prediction'] != '']
    target_minorities_blank = tm_pred.loc[tm_pred['prediction'] == '']
    generations = [[],[]]
    
    for i in range(target_minorities_blank.shape[0]):
      hitid = target_minorities_blank.iloc[i]['HITId']
      generations[0].append(hitid)
      generations[1].append([''])
    
    gpt_model = AutoModelForCausalLM.from_pretrained(args.implicit_knowledge_generator)
    gpt_model.cuda()
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.implicit_knowledge_generator, use_fast=True)
    
    tm_dataset = Dataset.from_pandas(target_minorities)
    mb_iter = MinibatchIterator(target_minorities, gpt_tokenizer)

    for i in tqdm(range(target_minorities.shape[0])):
      hitid = target_minorities.iloc[i]['HITId']
      target_minority = target_minorities.iloc[i]['prediction']
      
      generations[0].append(hitid)
      generations[1].append([])
      
      if args.k <= len(PROMPTS):
        sample_prompts = random.sample(PROMPTS, k=args.k)
      else:
        sample_prompts = random.choices(PROMPTS, k=args.k)
      
      for sample in sample_prompts:
        sample = 'The ' + target_minority + ' ' + sample
        input_ids = gpt_tokenizer(sample, return_tensors='pt').input_ids.cuda()
        outputs = gpt_model.generate(
            input_ids,
            #num_beams=5,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            max_length=16,
            #max_length=32,
            top_p=0.95,
            length_penalty=1.5,
            pad_token_id=gpt_tokenizer.eos_token_id,
            eos_token_id=gpt_tokenizer.eos_token_id,
        )
        generation = gpt_tokenizer.decode(outputs[0])
        generation = tokenize.sent_tokenize(generation)[0]
        generation = generation.strip()
        generations[1][-1].append(generation)
    return generations

def get_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file, train=True):
    df = df[['HITId','post']]
    df = df.drop_duplicates().reset_index(drop=True)

    tm_pred = run_target_minority_model(df, tm_pred_pickle_file, args)
    if not os.path.exists(generation_pickle_file):
      implicit_generations = run_implicit_generation(tm_pred, args)
      pickle.dump(implicit_generations, open(generation_pickle_file, 'wb'))
    else:
      implicit_generations = pickle.load(open(generation_pickle_file, 'rb'))
    
    implicit_generation_df = pd.DataFrame({"HITId": implicit_generations[0], "target": implicit_generations[1]})
    if train:
      implicit_generation_df = implicit_generation_df.explode(column="target", ignore_index=True)
    
    if df.HITId.dtype == np.int64:
      implicit_generation_df.HITId = implicit_generation_df.HITId.astype(np.int64)
    implicit_generation_df.set_index("HITId", inplace=True)
    
    df = df.join(implicit_generation_df, on="HITId")
    df.reset_index(inplace=True, drop=True)
    return df

def append_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file, train=True):
    df_stereotype = clean_target(df, target_col='targetStereotype', train=train)
    df_stereotype = df_stereotype[['HITId', 'target']]

    df = get_implicit_generations(df, args, tm_pred_pickle_file, generation_pickle_file)
    df = df.groupby(by=['HITId','post'], as_index=False).agg({'target': lambda col: SEP_TOKEN.join(col)})
    df.post = df.apply(lambda row: row['post'] + SEP_TOKEN + row['target'] if row['target'] != '' else row['post'], axis=1)
    df = df[['HITId', 'post']]

    df_stereotype.set_index("HITId", inplace=True)
    df = df.join(df_stereotype, on="HITId")
    df.reset_index(inplace=True, drop=True)
    
    return df

