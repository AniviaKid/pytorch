import os
from io import open
import torch
import re

class Dictionary(object):
    def __init__(self):
        self.word2idx = {} #dictionary
        self.idx2word = [] #list

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word) #将word插入list末尾
            self.word2idx[word] = len(self.idx2word) - 1 #向dictionary中插入一个新的键值对 word:idx, idx的数值是word在list中的索引
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train_wiki.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid_wiki.txt'))
        self.test = self.tokenize(os.path.join(path, 'test_wiki.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                #words = re.split(" |!|\?|\.", line)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class Test(object):

    human=[]

    def __init__(self):
        self.dictionary = Dictionary()
        self.data = self.tokenize( './data/wikitext-2/combined.csv')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        flag=True
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if flag:
                    flag=False
                else:
                    line = line.strip()
                    words = line.split(',')
                    self.human.append(words[len(words)-1])
                    words.pop()
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        flag=True
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                if flag:
                    flag=False
                else:
                    line = line.strip()
                    words = line.split(',')
                    words.pop()
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids