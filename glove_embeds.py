'''
This code was written by following the following tutorial:
Link: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

This script processes and generates GloVe embeddings
'''
# coding: utf-8

import pickle
from processData import Vocabulary
import numpy as np
import json
from scipy import misc
import bcolz

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir='glove.6B/6B.300.dat', mode='w')

with open('glove.6B/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir='glove.6B/6B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open('glove.6B/6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open('glove.6B/6B.300_idx.pkl', 'wb'))

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print('Loading vocab...')

vectors = bcolz.open('glove.6B/6B.300.dat')[:]
words = pickle.load(open('glove.6B/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open('glove.6B/6B.300_idx.pkl', 'rb'))

print('glove is loaded...')

glove = {w: vectors[word2idx[w]] for w in words}
matrix_len = len(vocab)
weights_matrix = np.zeros((matrix_len, 300))
words_found = 0

for i, word in enumerate(vocab.idx2word):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

pickle.dump(weights_matrix, open('glove.6B/glove_words.pkl', 'wb'), protocol=2)

print('weights_matrix is created')
