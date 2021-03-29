#!/usr/bin/env python
# coding: utf-8

import os
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import pickle

from pathlib import Path
out_dir= Path(os.getcwd() + r'\Models\Custom_NER_Model_1')
in_file = '/Datasets/Evaluation_Corpus_Tagged_NER.p'

def Eval(in_file,out_dir):
    # Evaluate the saved model
    print("Loading from", out_dir)
    ner_model = spacy.load(out_dir)
    
    with open (os.getcwd() +  r'%s'%in_file, 'rb') as fp:
        examples = pickle.load(fp)
    
    scorer = Scorer()
    try:
        for input_, annot in examples:
            doc_gold_text = ner_model.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot['entities'])
            pred_value = ner_model(input_)
            scorer.score(pred_value, gold)
    except Exception as e: print(e)

    print(scorer.scores)

Eval(in_file,out_dir)

