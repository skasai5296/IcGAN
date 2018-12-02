import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms

from dataset import CelebA, collate_fn, randomsample
from model import Generator

device = torch.device('cuda:2')
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(0.5, 0.5, 0.5))])

def see_image(args):
    celeba = CelebA('../../../local/CelebA/', 'img_align_celeba', 'list_attr_celeba.csv', transform=transform)
    gen = Generator(140)
    gen = gen.to(device)

    z = torch.randn((args.show_size, 100)).to(device)
    y = randomsample(celeba, args.show_size).to(device)

    eps = [i for i in range(1, args.max_epoch) if i % 5 == 0]
    for ep in eps:
        path = '../model/gen_epoch_{}.model'.format(ep)
        gen.load_state_dict(torch.load(path))

        attrnames = list(celeba.df.columns)[1:]
        out = gen(z, y).detach().cpu()
        vutils.save_image(out, '../out/sample_out_ep{}.png'.format(ep))
        multihot = y.detach().cpu().numpy()
        idxs = np.where(multihot == 1)
        a = []
        piv = 0
    with open('../out/sample_attr.txt', 'w') as f:
        for i, j in zip(*idxs):
            if i == piv:
                a.append(attrnames[j])
            else:
                f.write('{}th image: {}\n'.format(piv, a))
                a = []
                piv += 1

parser = argparse.ArgumentParser()
parser.add_argument('--show_size', type=int, default=64)
parser.add_argument('--max_epoch', type=int, default=20)
opt = parser.parse_args()

see_image(opt)
