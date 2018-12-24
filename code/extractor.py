import os
import argparse
import time
import io
import warnings
import random

import numpy as np
import nltk
import pickle

import torch
import torchvision.utils as vutils
from torchvision import transforms
from model import *

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Command():
    def __init__(self):
        self.flag = True
        while self.flag:
            print('Input a command: ')
            self.reset_command(input())
            if self.command in ['quit', 'exit', 'q', 'end']:
                self.flag = False
                break
            self.get_cleaned_tokens()
            if len(self.cleaned) < 3:
                warnings.warn('Command sequence is too short. Input a longer sequence')
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
                    words = random.sample(self.word, 2)
                    self.att = words[0]
                    self.des = words[1]
                    print('randomly choosing attributes', flush=True)
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
        self.descriptions_n = []

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
                self.descriptions_n.append(clean[2])
                cnt = i+1
        return self.attributes, self.descriptions, self.descriptions_n, cnt


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

parser = argparse.ArgumentParser()
parser.add_argument('--impath', type=str, default='../../../local/CelebA/img_align_celeba/202599.jpg')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--enc_y_path', type=str, default='../model/enc_y_epoch_30.model')
parser.add_argument('--enc_z_path', type=str, default='../model/enc_z_epoch_30.model')
parser.add_argument('--gen_path', type=str, default='../model/gen_epoch_200.model')
parser.add_argument('--save_path', type=str, default='../out/')
args = parser.parse_args()


'''load vectors from path'''
path = '''../corpus/crawl-300d-2M-subword.vec'''
picklepath = '''../corpus/embeddings.pkl'''
if not os.path.exists(picklepath):
    save_vectors(path, picklepath)
data = load_vectors(picklepath)

print(cosine_sim(data, 'black', 'gray'))

attrs = Attributes()
attributes, descriptions, descriptions_n, ftnum = attrs.get_attributes_descriptions('''attributes.txt''')
print(attributes, descriptions, descriptions_n)

a = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_y = Encoder(ftnum, for_y=True)
enc_z = Encoder(args.nz, for_y=False)
gen = Generator(ftnum+args.nz)
enc_y = enc_y.to(device)
enc_z = enc_z.to(device)
gen = gen.to(device)

enc_y.load_state_dict(torch.load(args.enc_y_path))
enc_z.load_state_dict(torch.load(args.enc_z_path))
gen.load_state_dict(torch.load(modelpath))

enc_y.eval()
enc_z.eval()
gen.eval()

im = Image.open(args.impath)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
im_trans = transform(im)
vutils.save_image(im_trans, args.save_path + 'before')
im_trans = torch.unsqueeze(im_trans, 0)

y = enc_y(im_trans)
z = enc_z(im_trans)

num = 1
while True:
    command = Command()
    if not command.flag:
        break
    word, pos = command.pos_tag_sequence()
    a.append(word)
    a.append(pos)
    att, des = command.get_attributes_descriptions()
    cnt = 0
    maxcos = -1
    minidx = []
    for i, attr in enumerate(attributes):
        cosine = cosine_sim(data, attr[i], att)
        print(cosine, flush=True)
        if cosine >= maxcos:
            minidx.append(i)
            minattr = attr[i]
            maxcos = cosine
    print('attribute {} to be modified'.format(minattr), flush=True)

    for i in minidx:
        cp = cosine_sim(data, desc, descriptions[i])
        cn = cosine_sim(data, desc, descriptions_n[i])
        if cp > cn+0.2:
            y[0, i] += 0.5
            print('increase', flush=True)
        elif cn > cp+0.2:
            y[0, i] -= 0.5
            print('decrease', flush=True)
        else: pass
        cnt += 1

    result = gen(z, y)
    vutils.save_image(result, args.save_path + 'after_{}'.format(num))
    print('saved image, {}th modification'.format(num))
    num += 1



if not command.flag:
    print(a)





