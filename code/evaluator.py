import argparse
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset import DeepFashion, CelebA, collate_fn, randomsample
from model import Generator, Discriminator, Encoder, init_weights, AttrEncoder, Resblock, Down, Up

def eval_im(model, imtensor, gt):
    cnt = 0
    allcnt = 0
    model.eval()
    out = model(imtensor)
    out[out >= 0.5] = 1
    out[out < 0.5] = 0
    cnt = int(torch.sum(torch.eq(out, gt)).detach().cpu())
    allcnt = int(torch.sum(torch.ones_like(gt)).detach().cpu())
    return cnt, allcnt

def eval(model, dataloader, device, early=True):
    cnt = 0
    allcnt = 0
    model.eval()
    for j, sample in enumerate(dataloader):
        im = sample['image'].to(device)
        t = sample['attributes'].to(device)

        out = model(im).detach()
        bs = im.size(0)
        out[out >= 0.5] = 1
        out[out < 0.5] = 0
        cnt = int(torch.sum(torch.eq(out, t)).detach().cpu())
        allcnt = int(torch.sum(torch.ones_like(t)).detach().cpu())
        if early:
            if j == 20: break
    return cnt, allcnt

def train(args):

    if args.use_tensorboard:
        writer = SummaryWriter(comment=args.log_name)

    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')

    # transforms applied
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # dataset and dataloader for training
    train_dataset = CelebA(args.root_dir, args.img_dir, args.ann_dir, transform=transform)
    test_dataset = CelebA(args.root_dir, args.img_dir, args.ann_dir, transform=transform, train=False)
    fsize = train_dataset.feature_size
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=args.show_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    '''
    dataloader returns dictionaries.
    sample : {'image' : (bs, 64, 64, 3), 'attributes' : (bs, att_size)}
    '''

    attnames = list(train_dataset.df.columns)[1:]

    # model, optimizer, criterion
    gen = Generator(in_c = args.nz + fsize)
    gen = gen.to(device)
    MODELPATH = '../model/gen_epoch_{}.model'.format(args.model_ep)
    gen.load_state_dict(torch.load(MODELPATH))

    enc_y = Encoder(fsize).to(device)
    MODELPATH = '../model/enc_y_epoch_{}.model'.format(args.enc_ep)
    enc_y.load_state_dict(torch.load(MODELPATH))
    enc_z = Encoder(args.nz, for_y=False).to(device)
    MODELPATH = '../model/enc_y_epoch_{}.model'.format(args.enc_ep)
    enc_z.load_state_dict(torch.load(MODELPATH))

    gen.eval()
    enc_y.eval()
    enc_z.eval()

    model = AttrEncoder(outdims=fsize).to(device)
    # initialize weights for encoders

    att_optim = optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas)
    criterion = nn.BCELoss().to(device)

    noise = torch.randn((args.batch_size, args.nz)).to(device)

    if args.use_tensorboard:
        writer.add_text("Text", "begin training, lr={}".format(args.learning_rate))
    print("begin training, lr={}".format(args.learning_rate), flush=True)
    stepcnt = 0

    for ep in range(args.num_epoch):
        for it, sample in enumerate(trainloader):
            elapsed = time.time()
            image = sample['image'].to(device)
            att = sample['attributes'].to(device)

            out = model(image)
            loss = criterion(out, att)
            z = enc_z(image)
            y = enc_y(image)
            out2 = gen(z, y)
            loss2 = criterion(out2, att)

            loss.backward()
            loss2.backward()
            att_optim.step()
            if it % args.log_every == (args.log_every - 1):
                if args.use_tensorboard:
                    writer.add_scalar('loss', loss, it+1)
                print("{}th iter \t loss: {:.8f} \t time per iter: {:.05f}s".format(it+1, loss.detach().cpu(), (time.time() - elapsed) / args.log_every), flush=True)

        cnt, allcnt = eval(model, testloader, device)
        print("-" * 50)
        print("epoch {} done. accuracy: {:.03f}%. num guessed: [{:05d}/{:05d}]".format(ep+1, cnt / allcnt * 100, cnt, allcnt))
        print("-" * 50, flush=True)

        if ep % args.save_every == (args.save_every - 1):
            torch.save(model.state_dict(), "../model/atteval_epoch_{}.model".format(ep+1))

    cnt, allcnt = eval(model, testloader, device, early=False)
    print("-" * 50)
    print("training done. final accuracy: {:.03f}% num guessed: [{:07d}/{:07d}]".format(ep+1, cnt / allcnt * 100, cnt, allcnt))
    print("-" * 50, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument('--model_ep', type=int, default=200)
    parser.add_argument('--enc_ep', type=int, default=50)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--show_size', type=int, default=64)
    parser.add_argument('--root_dir', type=str, default='../../../local/CelebA/')
    parser.add_argument('--img_dir', type=str, default='cropped')
    parser.add_argument('--ann_dir', type=str, default='list_attr_celeba.csv')
    parser.add_argument('--cuda_device', type=str, default='cuda')
    parser.add_argument('--nz', type=int, default=100)


    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
