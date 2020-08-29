"""

Basic, word-based BiLSTM model

@author: karina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import math
from utils import *


def prepare_emb(sent, tags, words_to_ix, tags_to_ix):
    """
    Maps each token to its embedding

    Parameters
    ----------
    sent : list
        sentence to process
    tags : list
        tag sequence for sentence
    words_to_ix : dict
        maps words to embedding indices
    tags_to_ix : dict
        maps tag to tag indices

    Returns
    -------
    torch.tensor
        2 dimensional tensor containing the indices for each word and
        the corresponding tag

    """
    w_idxs, tag_idxs = [], []
    for w, t in zip(sent, tags):
        if w.lower() in words_to_ix.keys():
            w_idxs.append(words_to_ix[w.lower()])
        else:
            # Map unknown words to the reserved embeddig vector
            w_idxs.append(words_to_ix['unk'])
                                      
        if t in tags_to_ix.keys():
            tag_idxs.append(tags_to_ix[t])
        else:
            # If an unknown tag is encountered during training, map the word to the 'O' tag
            tag_idxs.append(tags_to_ix['O'])
            
    return torch.tensor(w_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)

class WordModel(Module):
    def __init__(self, pretrained_embeddings, hidden_size, vocab_size, n_classes, dropout=0.0):
        super(WordModel, self).__init__()
        
        # Vocabulary size
        self.vocab_size = pretrained_embeddings.shape[0]
        # Embedding dimensionality
        self.embedding_size = pretrained_embeddings.shape[1]
        # Number of hidden units
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        
        # Dropout (optional)
        self.dropout = Dropout(p=dropout, inplace=False)
        
        # Hidden layer (300, 20)
        self.lstm = LSTM(self.embedding_size, self.hidden_size, num_layers=2)
        
        # Final prediction layer
        self.hidden2tag = Linear(self.hidden_size, n_classes)
    
    def forward(self, x):
        # Retrieve word embedding for input token
        emb = self.embedding(x)
        
        # Apply dropout
        dropout = self.dropout(emb)
        
        # Hidden layer
        h, _ = self.lstm(dropout.view(len(x), 1, -1))
        
        # Prediction
        pred = self.hidden2tag(h.view(len(x), -1))
        
        return F.log_softmax(pred, dim=1)
