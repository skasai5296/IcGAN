import os
import argparse
import time
import io
import glob
import warnings
import random

import numpy as np
import nltk
import pickle
from PIL import Image

import torch
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
from dataset import CelebA, collate_fn

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Command():
    def __init__(self):
        self.flag = True
        self.flag2 = True
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
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(word):
        return self.lemmatizer.lemmatize(word)

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
            if p == 'JJR':
                if self.word[cnt] == 'more':
                    if cnt != len(self.pos) - 1:
                        self.att = self.word[cnt+1]
                    else:
                        self.att = self.word[cnt-1]
                    self.des = 'more'
                    break
                elif self.word[cnt] == 'less':
                    if cnt != len(self.pos) - 1:
                        self.att = self.word[cnt+1]
                        self.des = 'less'
                        break
                else:
                    self.att = lemmatize(w)
                    self.des = 'more'
                    break
            elif p[:2] == 'NN':
                if cnt != len(self.pos) - 1:
                    if self.pos[cnt+1][:2] in ['JJ', 'RB', 'NN']:
                        self.att = w
                        self.des = word[cnt+1]
                        break
                elif self.pos[cnt-1][:2] == 'JJ':
                    self.att = word[cnt-1]
                    self.des = w
                    break
            elif p == 'JJ':
                self.att = lemmatize(w)
                self.des = 'more'
                break
            cnt += 1
            if cnt == len(self.pos) - 1:
                if self.att is None:
                    words = random.sample(self.word, 2)
                    self.att = words[0]
                    self.des = words[1]
                    print('randomly choosing attributes', flush=True)
                break
        print('using word "{}" as attribute, word "{}" as way of change'.format(self.att, self.des), flush=True)
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
parser.add_argument('--gen_path', type=str, default='../model/gen_epoch_60.model')
parser.add_argument('--save_path', type=str, default='../out/transfer/')
parser.add_argument('--show_size', type=int, default=64)
parser.add_argument('--root_dir', type=str, default='../../../local/CelebA/')
parser.add_argument('--img_dir', type=str, default='img_align_celeba')
parser.add_argument('--ann_dir', type=str, default='list_attr_celeba.csv')
args = parser.parse_args()

'''load vectors from path'''
path = '''../corpus/crawl-300d-2M-subword.vec'''
picklepath = '''../corpus/embeddings.pkl'''
if not os.path.exists(picklepath):
    save_vectors(path, picklepath)
data = load_vectors(picklepath)

attrs = Attributes()
attributes, descriptions, descriptions_n, ftnum = attrs.get_attributes_descriptions('''attributes.txt''')

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
gen.load_state_dict(torch.load(args.gen_path))

enc_y.eval()
enc_z.eval()
gen.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_dataset = CelebA(args.root_dir, args.img_dir, args.ann_dir, transform=transform, train=False)
testloader = DataLoader(test_dataset, batch_size=args.show_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
for i in testloader:
    sample = i
    break
im = sample['image']


grid = vutils.make_grid(im, normalize=True)
for i in range(100):
    pth = glob.glob(os.path.join(args.save_path, 'im_{}'.format(i)) + '*.jpg')
    if not pth:
        imcnt = i
        break
vutils.save_image(grid, args.save_path + 'im{}_0_original.jpg'.format(imcnt))

im = im.to(device)

y = enc_y(im)
z = enc_z(im)
del enc_z
del enc_y
torch.cuda.empty_cache()

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
    minidx = []
    cosine = np.full((ftnum, ), -1)
    maxcos = np.max(cosine)
    for i, attr in enumerate(attributes):
        if cosine[i] > maxcos-0.1:
            minidx.append(i)
            if cosine[i] == maxcos:
                minattr = attr
    print('attribute {} to be modified'.format(minattr), flush=True)

    for i in minidx:
        cp = cosine_sim(data, des, descriptions[i])
        cn = cosine_sim(data, des, descriptions_n[i])
        if cp > cn+0.1:
            y[:, i] = 1
            print('direction {}'.format(descriptions[i]), flush=True)
        elif cn > cp+0.1:
            y[:, i] = 0
            print('direction {}'.format(descriptions_n[i]), flush=True)
        else: 
            print('no change', flush=True)
        cnt += 1

    result = gen(z, y)
    result = result.detach().cpu()
    grid = vutils.make_grid(result, normalize=True)
    vutils.save_image(grid, args.save_path + 'im{}_{}_{}.jpg'.format(imcnt, num, command.command))
    print('saved image, {}th modification'.format(num))
    num += 1



if not command.flag:
    print(a)





