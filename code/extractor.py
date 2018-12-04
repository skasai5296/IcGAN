import os
import time
import io
import warnings

import numpy as np
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Command():
    def __init__(self):
        while True:
            print('Input a command: ')
            self.command = input()
            self.reset_command(self.command)
            get_cleaned_tokens(self)
            if len(self.cleaned) < 3:
                warnings.warn('Command sequence is too short. Input a longer sequence')
                print('reinput: ')
                self.command = input()
            else break

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
        for w, p in zip(self.word, self.pos):
            if cnt == len(pos) - 1:
                break
            elif p[:2] == 'NN':
                if p[cnt+1][:2] == 'JJ' or p[cnt+1][:2] == 'RB':
                    self.att = w
                    self.des = word[cnt+1]
                elif p[cnt-1][:2] == 'JJ':
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


def load_vectors(fname):
    print('loading vectors...', flush=True)
    before = time.time()
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = list(map(int, fin.readline().split()))
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    print('Done, {}s taken'.format(time.time() - before), flush=True)
    return data


'''load vectors from path'''
path = '''../corpus/crawl-300d-2M-subword.vec'''
data = load_vectors(path)
print(cosine_sim(data, 'black', 'gray'))

attrs = Attributes()
attributes, descriptions = attrs.get_attributes_descriptions('''attributes.txt''')

while True:
    command = Command()
    command.reset_command(input())
    word, pos = command.pos_tag_sequence()
    att, des = command.get_attributes_descriptions()
    cnt = 0
    maxcos = -1
    for attr, desc in zip(attributes, descriptions):
        cosine = ( cossim(data, attr, att) + cossim(data, desc, des) ) / 2
        if cosine > maxcos:
            minidx = cnt
            minattr = attr
            maxcos = cosine
    print('{}th attribute to be modified'.format(minidx+1))



maxtarword = comseg[maxidx]
maxattrword = attributes[maxattr][maxidx]

print('modify attribute {} having seen the word {} in the command, cossim = {}'.format(maxattrword, maxtarword, maxcos))
