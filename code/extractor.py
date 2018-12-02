import os
import time
import io
import warnings

import numpy as np
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Command():
    def __init__(self, command):
        self.command = command
        reset_command(self.command)
        get_cleaned_tokens(self)
        if len(self.cleaned) < 3:
            warnings.warn('Command sequence is too short. Input a longer sequence')

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
        print('could not calculate cosine_sim because of lack of word in dictionary')
        return None


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
print(cosine_sim(data, 'red', 'blue'))
print(cosine_sim(data, 'black', 'gray'))
command = Command('Change her hair red! %A& )$')
comseg = command.pos_tag_sequence()
print(comseg)

attrs = Attributes()
attributes, descriptions = attrs.get_attributes_descriptions('''attributes.txt''')

while True:
    command.reset_command(input())
    word, pos = command.pos_tag_sequence()
    cnt = 0
    for w, p in zip(word, pos):
        if cnt == len(pos):
            break
        if p[:2] == 'NN':
            if p[cnt+1][:2] == 'JJ' or p[cnt+1][:2] == 'RB':
                att = w
                des = word[cnt+1]
            elif p[cnt-1][:2] == 'JJ':
                att = word[cnt-1]
                des = w
        cnt += 1



maxcos = 1

cnt = 0
for attr, desc in zip(attributes, descriptions):
    cosine = cossim(data, attr, att)
    if cosine < maxcos:
        minidx = cnt
        minattr = attr
        maxcos = cosine
    cnt += 1
maxtarword = comseg[maxidx]
maxattrword = attributes[maxattr][maxidx]

print('modify attribute {} having seen the word {} in the command, cossim = {}'.format(maxattrword, maxtarword, maxcos))
