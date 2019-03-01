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
from model import Generator, Discriminator, Encoder, init_weights, AttrEncoder
from evaluator import eval, eval_im

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
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=args.show_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)

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
    enc_z = Encoder(args.nz, for_y=False).to(device)

    attr = AttrEncoder().to(device)
    ATTRPATH = '../model/atteval_epoch_{}.model'.format(args.attr_epoch)
    attr.load_state_dict(torch.load(ATTRPATH))
    attr.eval()

    # initialize weights for encoders
    enc_y.apply(init_weights)
    enc_z.apply(init_weights)

    enc_y_optim = optim.Adam(enc_y.parameters(), lr=args.learning_rate, betas=args.betas)
    enc_z_optim = optim.Adam(enc_z.parameters(), lr=args.learning_rate, betas=args.betas)
    criterion = nn.MSELoss()

    noise = torch.randn((args.batch_size, args.nz)).to(device)

    if args.use_tensorboard:
        writer.add_text("Text", "begin training, lr={}".format(args.learning_rate))
    print("begin training, lr={}".format(args.learning_rate), flush=True)
    stepcnt = 0
    gen.eval()
    

    for ep in range(args.num_epoch):

        enc_y.train()
        enc_z.train()

        YLoss = 0
        ZLoss = 0

        ittime = time.time()
        run_time = 0
        run_ittime = 0

        for it, sample in enumerate(trainloader):

            elapsed = time.time()

            x = sample['image'].to(device)
            y = sample['attributes'].to(device)

            '''training of attribute encoder'''
            # train on real images, target are real attributes
            enc_y.zero_grad()
            out = enc_y(x)
            l2loss = criterion(out, y)
            l2loss.backward()
            enc_y_optim.step()
            loss_y = l2loss.detach().cpu().item()

            '''training of identity encoder'''
            # train on fake images generated with fake labels, target are original identities
            enc_z.zero_grad()
            dist = Bernoulli(torch.Tensor([0.5]))
            y_fake = dist.sample(y.size()).squeeze().to(device)
            x_fake = gen(noise, y_fake)
            z_recon = enc_z(x_fake)
            l2loss2 = criterion(z_recon, noise)
            l2loss2.backward()
            enc_z_optim.step()
            loss_z = l2loss2.detach().cpu().item()

            YLoss += loss_y
            ZLoss += loss_z

            ittime_a = time.time() - ittime
            run_time += time.time() - elapsed
            run_ittime += ittime_a

            '''log the losses and images, get time of loop'''
            if it % args.log_every == (args.log_every - 1):
                if args.use_tensorboard:
                    writer.add_scalar('y loss', loss_y, stepcnt+1)
                    writer.add_scalar('z loss', loss_z, stepcnt+1)

                after = time.time()
                print("{}th iter\ty loss: {:.5f}\tz loss: {:.5f}\t{:.4f}s per step, {:.4f}s per iter".format(it+1, loss_y, loss_z, run_time / args.log_every, run_ittime / args.log_every), flush=True)
                run_time = 0
                run_ittime = 0
            ittime = time.time()

            stepcnt += 1

        print("epoch [{}/{}] done | y loss: {:.6f} \t z loss: {:.6f}]".format(ep+1, args.num_epoch, YLoss, ZLoss), flush=True)
        if args.use_tensorboard:
            writer.add_text("epoch loss", "epoch [{}/{}] done | y loss: {:.6f} \t z loss: {:.6f}]".format(ep+1, args.num_epoch, YLoss, ZLoss), ep+1)

        try:
            torch.save(enc_y.state_dict(), "../model/enc_y_epoch_{}.model".format(ep+1))
            torch.save(enc_z.state_dict(), "../model/enc_z_epoch_{}.model".format(ep+1))
        except OSError:
            print("failed to save model for epoch {}".format(ep+1))

        if ep % args.recon_every == (args.recon_every - 1):
            # reconstruction and attribute transfer of images
            enc_y.eval()
            enc_z.eval()
            SAVEPATH = '../out/enc_epoch-{}'.format(ep+1)
            if not os.path.exists(SAVEPATH):
                os.mkdir(SAVEPATH)

            for sample in testloader:
                im = sample['image']
                grid = vutils.make_grid(im, normalize=True)
                vutils.save_image(grid, os.path.join(SAVEPATH, '0original.png'))
                im = im.to(device)
                y = enc_y(im)
                z = enc_z(im)
                im = gen(z, y)
                y_h = attr(im)
                recon = im.cpu()
                grid = vutils.make_grid(recon, normalize=True)
                vutils.save_image(grid, os.path.join(SAVEPATH, '0recon.png'))

                CNT = 0
                ALLCNT = 0
                for idx in range(fsize):
                    fname = attnames[idx]
                    y_p_h = y_h.clone()
                    for i in range(args.show_size):
                        y_p_h[i, idx] = 0 if y_p_h[i, idx] == 1 else 1
                    out = gen(z, y_p_h)
                    trans = out.cpu()
                    grid2 = vutils.make_grid(trans, normalize=True)
                    vutils.save_image(grid2, os.path.join(SAVEPATH, '{}.png'.format(fname)))
                    cnt, allcnt = eval_im(attr, out, y_p_h)
                    CNT += cnt
                    ALLCNT += allcnt
                break
            print("epoch {} for encoder, acc: {:.03}%".format(ep+1, CNT / ALLCNT * 100), flush=True)



    print("end training")
    if args.use_tensorboard:
        writer.add_text("Text", "end training")
        writer.close()

def main():
    '''
    optional arguments

    log_name : name to append on end of log file on Tensorboard
    recon_every : conduct reconstruction experiment every ? epochs and save model
    log_every : log loss and image every ? iterations
    num_epoch : number of epochs to train
    learning_rate : learning rate of Adam optimizer
    betas : betas of Adam optimizer
    batch_size : batch size for training
    show_size : number of images to show
    root_dir : full path of data. should be parent for images and annotations
    img_dir : relative path of directory containing images
    ann_dir : relative path of file containing annotations of attributes.
    cuda_device : gpu device to use for training
    model_ep : epoch number of model to read as Generator and Discriminator
    nz : number of dimensions of noise vector
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--recon_every', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--show_size', type=int, default=64)
    parser.add_argument('--root_dir', type=str, default='../../../local/CelebA/')
    parser.add_argument('--img_dir', type=str, default='img_align_celeba')
    parser.add_argument('--ann_dir', type=str, default='list_attr_celeba.csv')
    parser.add_argument('--cuda_device', type=str, default='cuda')
    parser.add_argument('--model_ep', type=int, default=200)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--attr_epoch', type=int, default=8)


    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
