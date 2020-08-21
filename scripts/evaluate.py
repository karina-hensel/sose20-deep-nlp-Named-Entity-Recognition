#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluate a trained model on the test set data (work in progress)
Created on Fri Aug 21 23:57:45 2020

@author: karina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext import data

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import os
import math
from model.utils import *

# Data and hyperparameters
#embeddings_file = '../Data/embeddings/en/glove.6B.100d.bin'

embeddings_file = '../data/embeddings/en/GoogleNews-pruned2tweets.txt'
data_dir = '../data/conll2003/en/'
model_dir = '../model/'
model = 'm1.pkl'

#--- test ---
load_model(model_dir + model)

correct = 0
with torch.no_grad():
  for sent in data["test"]:
    input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
    # WRITE CODE HERE
    predicted, correct = 0.0, 0.0
    
    # Predict class with the highest probability
    if len(input_sent) > 0:
        predicted = torch.argmax(model(input_sent), dim=1)
        print(predicted)
        print(gold_tags)
        correct += torch.eq(predicted,gold_tags).item()
  
    if verbose:
      print('TEST DATA: %s, OUTPUT: %s, GOLD TAG: %d' % 
            (sent["TOKENS"], sent["NE"], predicted))
      
  print('test accuracy: %.2f' % (100.0 * correct / len(data["test"])))