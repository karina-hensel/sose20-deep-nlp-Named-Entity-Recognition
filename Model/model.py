#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:02:19 2020

@author: karina
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext import data
from torchtext.data import Field, BucketIterator
#from torchtext import data_iterator

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import os
import math
from utils import *

# Data and hyperparameters
#embeddings_file = '../Data/embeddings/en/glove.6B.100d.bin'
embeddings_file = '../Data/embeddings/en/GoogleNews-pruned2tweets.txt'
data_dir = '../Data/conll2003/en/'
model_dir = ''

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class Model(Module):
    def __init__(self, pretrained_embeddings, hidden_size, vocab_size, n_classes):
        super(Model, self).__init__()
        
        # Vocabulary size
        self.vocab_size = vocab_size#pretrained_embeddings.shape[0]
        # Embedding dimensionality
        self.embedding_size = pretrained_embeddings.shape[1]
        # Number of hidden units
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        #print(self.vocab_size, self.embedding_size)
        # Dropout
        #self.dropout = Dropout(p=0.5, inplace=False)
        # Hidden layer (300, 20)
        self.lstm = LSTM(self.embedding_size, self.hidden_size, num_layers=2)
        # Final prediction layer
        self.hidden2tag = Linear(self.hidden_size, n_classes)#, bias=True)
    
    def forward(self, x):
        # Retrieve word embedding for input token
        print(x)
        emb = self.embedding(x)
        # Apply dropout
        #dropout = self.dropout(emb)
        # Hidden layer
        h, _ = self.lstm(emb)#.unsqueeze(0))
        # = h.view(-1, h.shape[2])
        # Prediction
        pred = self.hidden2tag(h)
        
        return F.log_softmax(pred, dim=1) #NLLLoss(pred, dim=1)
    

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
    n_epochs = 2
    
    # Set up and initialize model
    print(len(word_to_idx))
    model = Model(pretrained_embeds, 100, len(word_to_idx), n_classes)
    loss_function = NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
    
    # Before training
    '''with torch.no_grad():
        inputs = prepare_sequence(data["test"][0]["TOKENS"], word_to_idx)
        tag_scores = model(inputs)
        print(tag_scores)
'''
    # Training loop
    for e in range(n_epochs+1):
        total_loss = 0
        for sent in data["train"][1:]:
            print(sent["TOKENS"])
            # (1) Set gradient to zero for new example: Set gradients to zero before pass
            model.zero_grad()
            
            # (2) Encode instance as sequence of indices
            #inp_sent = prepare_sequence(sent["TOKENS"][0], word_to_idx)
            #print(inp_sent[0])
            gold_tags = prepare_sequence(sent["NE"], TAG_INDICES)
            #print(inp_sent)
            #gold_tags = label_to_idx(sent["NE"])
            #pred_tags = []
            # (3) Predict tags
            pred_tags = []#model(inp_sent[0])
            # Process token by token (one-to-one)
            for i, word in enumerate(sent["TOKENS"]):
                print(word)
                if word in word_to_idx.keys():
                    word_embed = torch.tensor([word_to_idx[word]], dtype=torch.long)
                else:
                    word_embed = torch.tensor([0], dtype=torch.long)
                # (3) Forward pass
                print(word_embed.squeeze(1))
                pred_tags.append(model(word_embed))
            pred_tags = torch.tensor(pred_tags, dtype=torch.long)
            # (4) Compute loss and do backward step
            print(pred_tags)
            print(gold_tags)
            loss = loss_function(pred_tags, gold_tags)
            loss.backward()
          
            # (5) Optimize parameter values
            optimizer.step()
          
            # (6) Accumulate loss
            total_loss += loss
        if ((e+1) % report_every) == 0:
            print('epoch: %d, loss: %.4f' % (e, total_loss*100/len(data['train'])))