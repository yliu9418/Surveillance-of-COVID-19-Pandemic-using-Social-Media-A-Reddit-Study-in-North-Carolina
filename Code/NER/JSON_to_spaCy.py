#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Convert json file to spaCy format.

import plac
import logging
import argparse
import sys
import os
import json
import pickle

#@plac.annotations(input_file=("Input file", "option", "i", str), output_file=("Output file", "option", "o", str))

def json_to_spacy(input_file, output_file):
    try:
        training_data = []
        lines=[]
        with open(os.getcwd() +  r'%s'%input_file, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        print(training_data)

        with open(os.getcwd() +  r'%s'%output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None

in_file = '/Datasets/Evaluation_Corpus_Tagged_NER.json'
out_file = '/Datasets/Evaluation_Corpus_Tagged_NER.p'

# Run once for Training data, and again for Evaluation data
# Change input/output directories/filenames above accordingly
json_to_spacy(in_file, out_file)


# In[ ]:




