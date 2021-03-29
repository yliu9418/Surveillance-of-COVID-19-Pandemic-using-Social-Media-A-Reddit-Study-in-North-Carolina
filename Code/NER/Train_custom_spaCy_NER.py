#!/usr/bin/env python
# coding: utf-8

import os
import spacy
import pickle

# New label(s) to add
LABEL = ['PPE', 'DIST','TEST','SYM','DIT']
in_file = '/Datasets/Training_Corpus_Tagged_NER.p'
model = None
new_model_name='Custom_NER_Model_1' #make sure to change when training another new
n_iter=30

# Load training examples in the required format
with open (os.getcwd() +  r'%s'%in_file, 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

if model is not None:
    nlp = spacy.load(model)  # load existing spacy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  # create blank Language class
    print("Created blank 'en' model")
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

# Add the new label to ner
for i in LABEL:
    ner.add_label(i)

# Resume training
if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()
    move_names = list(ner.move_names)

# List of pipes you want to train
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Importing requirements
from spacy.util import minibatch, compounding
import random

# Begin training by disabling other pipeline components
with nlp.disable_pipes(*other_pipes):  # since only training NER
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                       losses=losses)
        print('Losses', losses)

# Output directory
from pathlib import Path
output_dir = Path('\Models')

# Saving the model to the output directory
if not output_dir.exists():
  output_dir.mkdir()
nlp.meta['name'] = 'cus_ner1'  # rename model
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

