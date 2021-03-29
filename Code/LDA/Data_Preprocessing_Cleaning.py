#!/usr/bin/env python
# coding: utf-8

import re, string, unicodedata
import spacy
from nltk.stem.porter import PorterStemmer
import pandas as pd
import os
import numpy as np
import re

subred_list = ['Charlotte', 'raleigh','gso','NorthCarolina']

from spacy.lang.en import English
nlp = English()

all_stopwords = nlp.Defaults.stop_words
all_stopwords -= {"from", "of"} #removed from stopwords for NER
all_stopwords |= {"like", "think","lol"} #added to stopwords to improve LDA results

def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())

def lemme_remove_stop_words(text):
    doc = nlp(str(text))
    result = ''
    for token in doc:
        if token.text in all_stopwords:
            continue
        if token.is_digit:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result += token.lemma_ + ' '
  
    return result

def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)

def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", text)

#Didn't use the stem_words function, retained for those that may
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s%]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    return words

def convert2words(text):
    doc = nlp(str(text))
    tokens = [token.orth_ for token in doc]
    return convert_list_to_string(normalize(tokens))

def cleanedData(sName):
    df = pd.read_csv(os.getcwd() + r'/reddit_data/%s.csv'%sName, encoding = 'utf-8')
    df['cleaned'] = df['combined'].astype('U').values
    df.cleaned = df['cleaned'].apply(str).astype('U').values
    df.cleaned = df['cleaned'].apply(remove_URL).astype('U').values
    df.cleaned = df['cleaned'].apply(lemme_remove_stop_words).astype('U').values
    df.cleaned = df['cleaned'].apply(convert2words).astype('U').values
    df.cleaned = df['cleaned'].apply(remove_whitespace).astype('U').values

    df.to_csv(os.getcwd() + r'/reddit_data/Cleaned/%s.csv'%sName, index=False, encoding="utf-8")

for i in subred_list:
    cleanedData(i)

