#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, pos_tag
from nltk.tokenize import word_tokenize

subred_list = ['Charlotte','raleigh','gso','NorthCarolina']

def getEnts(sName):
    df = pd.read_csv(os.getcwd() + r'\NER_Results\%s.csv'%sName, encoding = 'utf-8')
    df1= df.groupby(['Entity Label'])['Entity Label'].count().to_frame(name = '# of Ent Types').reset_index()
    df1.to_csv(os.getcwd() + r'\NER_Results\%s_Num_of_tags.csv'%sName, index=False, encoding = 'utf-8')

for sName in subred_list:
    getEnts(sName)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # makes if-statement more simple

def lemme_sentences(sentence):
    lemmatizer = WordNetLemmatizer()
    lemme_tokens = []
    tagged = pos_tag(sentence.split())
    print(tagged)
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:# no tag supplied in the case of None
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        lemme_tokens.append(lemma)
    return ' '.join(lemme_tokens)

def getEntLabel_Data(sName):
    df = pd.read_csv(os.getcwd() + r'\NER_Results\%s.csv'%sName, encoding = 'utf-8')
    df['Entity Name'] =  df['Entity Name'].apply(lemme_sentences).astype('U').values
    df1= df.groupby(['Entity Name','Entity Label'])['Entity Name'].count().to_frame(name = '# of Entities').reset_index()
    total = df1['# of Entities'].sum()
    print(df1)
    df1.to_csv(os.getcwd() + r'\NER_Results\%s_Entities.csv'%sName, index=False, encoding = 'utf-8')
    print('Total entities = %s\n'%total)

for sName in subred_list:
    getEntLabel_Data(sName)

