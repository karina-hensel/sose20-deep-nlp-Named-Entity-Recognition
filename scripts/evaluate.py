"""

Evaluate a trained model on the test set data

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
import argparse
from utils import *
from word_model import WordModel, prepare_emb
from train import *


if __name__=='__main__':
    # Set up to parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the test set data')
    
    parser.add_argument('Embedding', metavar='EMB_FILE', type=str,
                   help='Path to word embedding file')
    parser.add_argument('Data_file', metavar='DATA_FILE', type=str,
                   help='Path to CONLL data set')
    parser.add_argument('Model_file', metavar='MODEL_FILE', type=str,
                   help='Name of the file to load the pretrained model from')
    
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load data and model
    dataset = read_conll_datasets(args.Data_file)
    print(args.Model_file)
    model = load_model(str(args.Model_file), device)
    emb_file = args.Embedding
    
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(emb_file, encoding='utf8')
    pretrained_embeds = gensim_embeds.vectors 
    
    # To convert words in the input to indices of the embeddings matrix
    word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}
        
    correct, num_words, f1_scores, num_sents = 0, 0, 0, 0
    with torch.no_grad():
        for sent in dataset["test"]:
            
            num_words += len(sent["TOKENS"])
            input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
            
            predicted, cor = 0.0, 0.0
    
            # Predict class with the highest probability
            if len(input_sent) > 0:
                num_sents += 1
                predicted = torch.argmax(model(input_sent.to(device)), dim=1)
                
                correct += torch.sum(torch.eq(predicted.to(device),gold_tags.to(device)))
                f1_scores += f1_score(predicted.cpu(), gold_tags.cpu(), average='weighted')
                
    if model.dropout == 0.0:
        print('Evaluation of the word-based BiLSTM without dropout:')
    else:
        print('Evaluation of the word-based BiLSTM with dropout:')
    print('----------------------------------------------------------')
    print('Test set accuracy: %.2f' % (100.0 * correct / num_words))
    print('Test set f1-score: %.2f' % (100.0 * f1_scores / num_sents))
