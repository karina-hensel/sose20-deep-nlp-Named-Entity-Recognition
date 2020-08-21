#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Basic, word-based BiLSTM model

Created on Tue Aug  4 18:02:19 2020

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
from utils import *

# Data and hyperparameters
#embeddings_file = '../Data/embeddings/en/glove.6B.100d.bin'

embeddings_file = '../data/embeddings/en/GoogleNews-pruned2tweets.txt'
data_dir = '../data/conll2003/en/'
model_dir = '../models/'

def prepare_emb(sent, tags, words_to_ix, tags_to_ix):
    w_idxs, tag_idxs = [], []
    for w, t in zip(sent, tags):
        if w.lower() in words_to_ix.keys():
            w_idxs.append(words_to_ix[w.lower()])# = [to_ix[w] for w in seq if w in to_ix.keys()]
        else:
            # Use 'Frock' as dummy for unknown words (only temporary solution)
            w_idxs.append(words_to_ix['frock'])
                                      
        if t in tags_to_ix.keys():
            tag_idxs.append(tags_to_ix[t])
        else:
            tag_idxs.append(tags_to_ix['O'])
            
    return torch.tensor(w_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)

class Model(Module):
    def __init__(self, pretrained_embeddings, hidden_size, vocab_size, n_classes):
        super(Model, self).__init__()
        
        # Vocabulary size
        self.vocab_size = pretrained_embeddings.shape[0]
        # Embedding dimensionality
        self.embedding_size = pretrained_embeddings.shape[1]
        # Number of hidden units
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        
        # Dropout
        #self.dropout = Dropout(p=0.5, inplace=False)
        # Hidden layer (300, 20)
        self.lstm = LSTM(self.embedding_size, self.hidden_size, num_layers=2)
        # Final prediction layer
        self.hidden2tag = Linear(self.hidden_size, n_classes)#, bias=True)
    
    def forward(self, x):
        # Retrieve word embedding for input token
        emb = self.embedding(x)
        # Apply dropout
        #dropout = self.dropout(emb)
        # Hidden layer
        h, _ = self.lstm(emb.view(len(x), 1, -1))
        # Prediction
        pred = self.hidden2tag(h.view(len(x), -1))
        
        return F.log_softmax(pred, dim=1)
    

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
    save_model(model, model_dir + 'm1.pkl')
    
    # Load model from file
    load_model(model_dir + 'm1.pkl')
    '''correct = 0
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
          
      print('test accuracy: %.2f' % (100.0 * correct / len(data["test"])))'''