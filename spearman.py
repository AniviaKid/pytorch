import torch
import torch.nn as nn
import math
import argparse
import model
import time

import os
from io import open

import scipy.stats as stats


class Dictionary(object):
    def __init__(self):
        self.word2idx = {} #dictionary
        self.idx2word = [] #list

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word) 
            self.word2idx[word] = len(self.idx2word) - 1 
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus:

    human=[]
    data=[]

    def __init__(self, data,human): #input preprocessed data by Load_data, data is a list, every elememt in this list has 2 words (word1 and word2)
        self.dictionary = Dictionary()
        self.data=data
        self.human=human
        self.train = self.tokenize()

    def get_train(self):
        return self.train

    def tokenize(self):
        """Tokenizes a text file."""
        # Add words to the dictionary
        for i in self.data:
            for j in i:
                self.dictionary.add_word(j)
                
        # Tokenize file content
        idss=[]
        for i in self.data:
            ids=[]
            for j in i:
                ids.append(self.dictionary.word2idx[j])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)
        return ids



def Load_data(input_path):
    assert os.path.exists(input_path)
    flag=True #use it to pass the first line in the dataset
    data=[]
    human=[]
    with open(input_path, 'r', encoding="utf8") as f:
        for line in f:
            if flag:
                flag=False
            else:
                line = line.strip()
                words = line.split(',')
                human.append(words[len(words)-1])
                words.pop()
                data.append(words)
    return data,human

