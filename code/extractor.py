import os
import time
import io
import warnings
import random

import numpy as np
import nltk
import pickle

import torch
from model import *

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Command():
    def __init__(self):
        while True:
            print('Input a command: ')
            self.reset_command(input())
            self.get_cleaned_tokens()
            if len(self.cleaned) < 3:
                warnings.warn('Command sequence is too short. Input a longer sequence')
                print('reinput: ')
                self.command = input()
            else:
                break

    def reset_command(self, command):
        self.command = command

    def get_cleaned_tokens(self):
        tokens = self.command.rstrip().split()
        cleaned = [token.strip(' ./<>!?|\'"(){}[]@;:_-=^~`*+$%#&') for token in tokens]
        self.cleaned = [token.lower() for token in cleaned if token != '']

    def pos_tag_sequence(self):
        self.get_cleaned_tokens()
        self.postag = nltk.pos_tag(self.cleaned)
        self.word = []
        self.pos = []
        for i in self.postag:
            self.word.append(i[0])
            self.pos.append(i[1])
        return self.word, self.pos

    def get_attributes_descriptions(self):
        cnt = 0
        self.att = None
        self.des = None
        for w, p in zip(self.word, self.pos):
            if cnt == len(self.pos) - 1:
                if self.att is None:
                    idx = random.randint(0, len(self.word) - 2)
                    self.att = self.word[idx]
                    self.des = self.word[idx+1]
                    print('randomly choosing attributes', flush=True)
                else:
                    print('using word "{}" as attribute, word "{}" as way of change'.format(self.att, self.des), flush=True)
                break
            elif p[:2] == 'NN':
                if self.pos[cnt+1][:2] in ['JJ', 'RB', 'NN']:
                    self.att = w
                    self.des = word[cnt+1]
                elif self.pos[cnt-1][:2] == 'JJ':
                    self.att = word[cnt-1]
                    self.des = w
            cnt += 1
        return self.att, self.des


class Attributes():
    def __init__(self):
        self.attributes = []
        self.descriptions = []

    def get_cleaned_tokens(self, line):
        tokens = line.rstrip().split()
        cleaned = [token.strip(' ./<>!?|\'"(){}[]@;:_-=^~`*+$%#&') for token in tokens]
        self.cleaned = [token.lower() for token in cleaned if token != '']
        return self.cleaned

    def get_attributes_descriptions(self, path):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                clean = self.get_cleaned_tokens(line)
                self.attributes.append(clean[0])
                self.descriptions.append(clean[1])
        return self.attributes, self.descriptions


def cosine_sim(embedding, w1, w2):
    try:
        vec1 = np.array(embedding[w1])
        vec2 = np.array(embedding[w2])
        sim = np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return sim
    except LookupError:
        return -1

def save_vectors(fname, picklepath):
    before = time.time()
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = list(map(int, fin.readline().split()))
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    max_bytes = 2**31 - 1
    bindata = pickle.dumps(data)
    binsize = len(bindata)
    with open(picklepath, 'wb') as f:
        for idx in range(0, binsize, max_bytes):
            f.write(bindata[idx:idx+max_bytes])
    print('Done pickling data, {}s taken'.format(time.time() - before), flush=True)
    return binsize

def load_vectors(picklepath):
    print('loading vectors...', flush=True)
    before = time.time()
    bindata = bytearray()
    binsize = os.path.getsize(picklepath)
    max_bytes = 2**31 - 1
    with open(picklepath, 'rb') as f:
        for _ in range(0, binsize, max_bytes):
            bindata += f.read(max_bytes)
    data = pickle.loads(bindata)
    print('Done loading pickled data, {}s taken'.format(time.time() - before), flush=True)
    return data

'''load vectors from path'''
path = '''../corpus/crawl-300d-2M-subword.vec'''
picklepath = '''../corpus/embeddings.pkl'''
if not os.path.exists(picklepath):
    save_vectors(path, picklepath)
data = load_vectors(picklepath)

print(cosine_sim(data, 'black', 'gray'))

attrs = Attributes()
attributes, descriptions = attrs.get_attributes_descriptions('''attributes.txt''')
print(attributes, descriptions)

while True:
    command = Command()
    word, pos = command.pos_tag_sequence()
    print(pos)
    att, des = command.get_attributes_descriptions()
    cnt = 0
    maxcos = -1
    for attr, desc in zip(attributes, descriptions):
        cosine = ( cosine_sim(data, attr, att) + cosine_sim(data, desc, des) ) / 2
        print(cosine, flush=True)
        if cosine >= maxcos:
            minidx = cnt
            minattr = attr
            maxcos = cosine
        cnt += 1
    print('{}th attribute to be modified'.format(minidx+1), flush=True)
    print(cosine_sim(data, 'make', 'more'))
    print(cosine_sim(data, 'make', 'less'))




maxtarword = comseg[maxidx]
maxattrword = attributes[maxattr][maxidx]

print('modify attribute {} having seen the word {} in the command, cossim = {}'.format(maxattrword, maxtarword, maxcos))
