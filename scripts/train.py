"""

Train a word-based BiLSTM for NER

@author: karina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim
from sklearn.metrics import f1_score

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import argparse
from utils import *
from word_model import WordModel, prepare_emb

if __name__=='__main__':
    # Set up to parse command line arguments
    parser = argparse.ArgumentParser(description='Train a word-based BiLSTM for NER')
    
    parser.add_argument('Embedding', metavar='EMB_FILE', type=str,
                   help='Path to word embedding file')
    parser.add_argument('Data_file', metavar='DATA_FILE', type=str,
                   help='Path to CONLL data set')
    parser.add_argument('Number_epochs', metavar='E', type=int,
                   help='Number of epochs')
    parser.add_argument('Dropout_rate', metavar='DROP', type=float,
                   help='Dropout rate')
    parser.add_argument('Model_file', metavar='MODEL_FILE', type=str,
                   help='Name of the file to save the trained model')
    
    args = parser.parse_args()
    
    # Check that correct number of arguments was given
   
        
    EMB_FILE = str(args.Embedding)
    DATA_DIR = str(args.Data_file)
    EPOCHS = args.Number_epochs
    DROPOUT = args.Dropout_rate
    MODEL_FILE = str(args.Model_file)
    
    # Load data
    data = read_conll_datasets(DATA_DIR)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(EMB_FILE, encoding='utf8')
    pretrained_embeds = gensim_embeds.vectors 
    
    # To convert words in the input to indices of the embeddings matrix
    word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}
    
    # Set hyperparameters
    # Number of output classes (9)
    n_classes = len(TAG_INDICES)
    n_epochs = EPOCHS
    p = DROPOUT
    report_every = 1
    
    # Set up and initialize model
    model = WordModel(pretrained_embeds, 100, len(word_to_idx), n_classes, p)
    loss_function = NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Training loop
    for e in range(n_epochs+1):
        total_loss = 0
        for sent in data["train"]:
            
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
    save_model(model, MODEL_FILE, optimizer, loss_function)
    
    
