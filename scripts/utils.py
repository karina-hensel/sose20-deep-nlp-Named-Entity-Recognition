import torch
import torch as nn
import codecs

""" Utility functions to load the data sets and preprocess the input """

TAG_INDICES = {'I-PER':0, 'B-PER':1, 'I-LOC':2, 'B-LOC':3, 'I-ORG':4, 'B-ORG':5, 'I-MISC':6, 'B-MISC':7, 'O':8}

def load_sentences(filepath):
    """
    Load sentences (separated by newlines) from dataset

    Parameters
    ----------
    filepath : str
        path to corpus file

    Returns
    -------
    List of sentences represented as dictionaries

    """
    
    sentences, tok, pos, chunk, ne = [], [], [], [], []

    with open(filepath, 'r') as f:
        for line in f.readlines():
           if line == ('-DOCSTART- -X- -X- O\n') or line == '\n':
               #pass
           #if line=='\n':
               # Sentence as a sequence of tokens, POS, chunk and NE tags
               sentence = dict({'TOKENS' : [], 'POS' : [], 'CHUNK_TAG' : [], 'NE' : [], 'SEQ' : []})
               sentence['TOKENS'] = tok
               sentence['POS'] = pos
               sentence['CHUNK_TAG'] = chunk
               sentence['NE'] = ne
               
               # Once a sentence is processed append it to the list of sentences
               sentences.append(sentence)
               
               # Reset sentence information
               tok = []
               pos= []
               chunk = []
               ne = []
           else:
               l = line.split(' ')
               
               # Append info for next word
               tok.append(l[0])
               pos.append(l[1])
               chunk.append(l[2])
               ne.append(l[3].strip('\n'))
    
    return sentences


def word_char_dicts(wor):
    """
    Create a dictionary of all words in the file:
        
        sentences - list of sentence dictionaries.

    Returns - set of unique tokens

    """
    
    words = set()
    
    for s in wor:
        words.update(s['TOKENS'])
    
    # Create character dictionary
    separator = ', '
    chars = set(separator.join(words))
    words.update('PAD')
    words.update('UNK')
    return words, chars

def ne_labels_dict(sentences):
    """
    Create a dictionary of all NE labels in the file

    sentences - list of sentence dictionaries.

    Returns - Set of unique NE labels

    """
    
    labels = set()
    
    for s in sentences:
        labels.update(s['NE'])
    
    return labels

#def build_vocab(sents, labels):
    
def read_conll_datasets(data_dir):
    data = {}
    for data_set in ["train","test","valid"]:
        data[data_set] = load_sentences("%s/%s.txt" % (data_dir,data_set)) 
        
    #words = word_char_dicts(data["train"])[0]
    return data#, words

def save_model(model, name, optimizer, loss):
    """
    Print evaluation of saved model

    Parameters
    ----------
    model : Model
        BiLSTM model loaded from file.
    name : String
        File name.
    """
    torch.save(model, name +'.pt')
    torch.save(model.state_dict(), name + '_2.pt')
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss.state_dict()
}, name + '_state_dict.pt')