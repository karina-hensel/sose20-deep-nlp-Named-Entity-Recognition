#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train a BiLSTM for NER (work in progress)

Created on Fri Aug 21 23:42:55 2020

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
from model.model import Model

# Data and hyperparameters
#embeddings_file = '../Data/embeddings/en/glove.6B.100d.bin'
#cwd = os.path.dirname(os.getcwd() + '/..')
embeddings_file = '../data/embeddings/en/GoogleNews-pruned2tweets.txt'
data_dir = '../data/conll2003/en/'
model_dir = '../model/'
model = 'm1.pkl'

if __name__=='__main__':
    # Load data
    data = read_conll_datasets(data_dir)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, encoding='utf8')
    pretrained_embeds = gensim_embeds.vectors 
    
    # To convert words in the input to indices of the embeddings matrix:
    word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}
    
    # Hyperparameters
    # Number of output classes (9)
    n_classes = len(TAG_INDICES)
    # Epochs
    n_epochs = 1
    # Batch size (currently not used)
    batch_size = 32
    report_every = 1
    verbose = True
    
    # Set up and initialize model
    model = Model(pretrained_embeds, 100, len(word_to_idx), n_classes)
    loss_function = NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
    
    # Training loop
    for e in range(n_epochs+1):
        total_loss = 0
        for sent in data["train"][5:10]:
            
            # (1) Set gradient to zero for new example: Set gradients to zero before pass
            model.zero_grad()
            
            # (2) Encode sentence and tag sequence as sequences of indices
            input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
            
            # (3) Predict tags (sentence by sentence)
            if len(input_sent) > 0:
                pred_scores = model(input_sent)
                
                # (4) Compute loss and do backward step
                loss = loss_function(pred_scores, gold_tags)
                loss.backward()
              
                # (5) Optimize parameter values
                optimizer.step()
          
                # (6) Accumulate loss
                total_loss += loss
        if ((e+1) % report_every) == 0:
            print('epoch: %d, loss: %.4f' % (e, total_loss*100/len(data['train'])))
            
    # Save the trained model
    save_model(model, 'm1.pkl')