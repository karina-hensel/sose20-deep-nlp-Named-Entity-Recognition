import torch
import torch as nn
import codecs

""" Utility functions to load the data sets and preprocess the input """

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
           if line == ('\n'):
               # Sentence as a sequence of tokens, POS, chunk and NE tags
               sentence = dict({'TOKENS' : [], 'POS' : [], 'CHUNK_TAG' : [], 'NE' : []})
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
    Create a dictionary of all words in the file

    Parameters
    ----------
    sentences : list
        List of sentence dictionaries.

    Returns
    -------
    Set of unique tokens

    """
    
    words = set()
    
    for s in wor:
        words.update(s['TOKENS'])
    
    # Create character dictionary
    separator = ', '
    chars = set(separator.join(words))
    
    return words, chars

def ne_labels_dict(sentences):
    """
    Create a dictionary of all NE labels in the file

    Parameters
    ----------
    sentences : list
        List of sentence dictionaries.

    Returns
    -------
    Set of unique NE labels

    """
    
    labels = set()
    
    for s in sentences:
        labels.update(s['NE'])
    
    return labels

# =========================
# Test
sents = load_sentences('../Data/conll2003/en/test.txt')
#print(sents)
tokens = word_char_dicts(sents)
labels = ne_labels_dict(sents)
print(labels)  