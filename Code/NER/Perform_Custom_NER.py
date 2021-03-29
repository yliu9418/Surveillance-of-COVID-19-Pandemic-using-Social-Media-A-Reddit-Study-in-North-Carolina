#!/usr/bin/env python
# coding: utf-8

import spacy
import os
from pathlib import Path
import pandas as pd
import numpy as np

input_dir= Path(os.getcwd() + r'\Models\Custom_NER_Model_1')

print("Loading from", input_dir)
nlp = spacy.load(input_dir)

subred_list = ['Charlotte','raleigh','gso','NorthCarolina']

def getText(sName):
    df = pd.read_csv(os.getcwd() + r'/reddit_data/Cleaned/%s.csv'%sName, encoding = 'utf-8')
    df.cleaned.apply(str).astype('U').values
    df.cleaned.apply(perform_NER)

def perform_NER(text):
    nerData = []
    doc = nlp(str(text))
    file = os.getcwd() + r'\NER_Results\%s.csv'%sName #make sure directory already exist
    from spacy import displacy
    
    colors = {'PPE': '#4cf45f', 'DIST': '#99c2ff','TEST': '#ffb3b3', 'SYM': '#f0b3ff', 'DIT': '#ffffb3'}
    options = {'ents': ['PPE', 'DIST','TEST','SYM','DIT'], 'colors':colors}
    displacy.render(doc,style='ent',jupyter=True, options=options)
    print('End of file %s.csv\n\n'%sName)
    
    for entity in doc.ents:
        nerData.append([entity,entity.label_])
    column_names = ['Entity Name', 'Entity Label']
    df = pd.DataFrame(nerData,columns = column_names)
    if not os.path.isfile(file):
        df.to_csv(file, header='column_names', index = False, encoding = 'utf-8')
    else: # else it exists so append without writing the header

for sName in subred_list:
    getText(sName)

