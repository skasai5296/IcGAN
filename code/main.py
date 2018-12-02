import argparse
import os
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
from model import Generator, Discriminator, init_weights


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
    fsize = train_dataset.feature_size
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    '''
    dataloader returns dictionaries.
    sample : {'image' : (bs, H, W, 3), 'attributes' : (bs, fsize)}
    '''

    attnames = list(train_dataset.df.columns)[1:]

    # model, optimizer, criterion
    gen = Generator(in_c = args.nz + fsize)
    gen = gen.to(device)
    dis = Discriminator(ny=fsize)
    dis = dis.to(device)

    # initialize weights for conv layers
    gen.apply(init_weights)
    dis.apply(init_weights)

    gen_optim = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=args.betas)
    dis_optim = optim.Adam(dis.parameters(), lr=args.learning_rate, betas=args.betas)
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.show_size, args.nz).to(device)
    # fixed_label = torch.zeros(args.show_size, fsize).to(device)
    fixed_label = randomsample(train_dataset, args.show_size).to(device)

    label = torch.empty(args.batch_size).to(device)
    noise = torch.empty(args.batch_size, args.nz).to(device)

    if not os.path.exists('../model'):
        os.mkdir('../model')
    if not os.path.exists('../out'):
        os.mkdir('../out')

    if args.use_tensorboard:
        writer.add_text("Text", "begin training, lr={}".format(args.learning_rate))
    print("begin training, lr={}".format(args.learning_rate), flush=True)
    stepcnt = 0
    gen.train()
    dis.train()

    real_tar = 0.8
    fake_tar = 0.2

    for ep in range(args.num_epoch):

        Dloss = 0
        Gloss = 0

        for it, sample in enumerate(trainloader):

            elapsed = time.time()

            x = sample['image'].to(device)
            y = sample['attributes'].to(device)

            '''training of Discriminator'''
            # train on real images and real attributes, target are ones (real)
            dis.zero_grad()
            real_label = torch.full_like(label, real_tar)
            x_real = x.clone().detach()
            y_real = y.clone().detach()
            out = dis(x_real, y_real)
            real_dis_loss = criterion(out, real_label)
            real_dis_loss.backward()
            D_x = out.mean().detach()

            # train on real images and fake attributes, target are zeros (fake)
            fake_label = torch.full_like(label, fake_tar)
            dist = Bernoulli(torch.Tensor([0.5]))
            x_real = x.clone().detach()
            y_fake = dist.sample(y.size()).squeeze().to(device)
            out = dis(x_real, y_fake)
            fake_dis_loss = 0.5 * criterion(out, fake_label)
            fake_dis_loss.backward()
            D_x2 = out.mean().detach()

            # train on fake images and real attributes, target are zeros (fake)
            fake_label = torch.full_like(label, fake_tar)
            z = torch.randn_like(noise)
            y_real = y.clone().detach()
            x_fake = gen(z, y_real)
            out = dis(x_fake, y_real)
            fake_dis_loss2 = 0.5 * criterion(out, fake_label)
            fake_dis_loss2.backward()
            D_G_z1 = out.mean().detach()

            dis_optim.step()
            errD = (real_dis_loss + fake_dis_loss + fake_dis_loss2) / 2

            '''training of Generator'''
            # train on fake images and real attributes, target are ones (real)
            gen.zero_grad()
            y_real = randomsample(train_dataset, args.batch_size).to(device)
            z = torch.randn_like(noise)
            x_prime = gen(z, y_real)
            out = dis(x_prime, y_real)
            real_label = torch.full_like(label, real_tar)
            gen_loss = criterion(out, real_label)
            gen_loss.backward()
            D_G_z2 = out.mean().detach()

            gen_optim.step()
            errG = gen_loss

            dloss = errD.item()
            gloss = errG.item()
            Dloss += dloss
            Gloss += gloss

            '''log the losses and images, get time of loop'''
            if it % args.log_every == (args.log_every - 1):

                if args.use_tensorboard:
                    writer.add_scalar("D_x", D_x, stepcnt)
                    writer.add_scalar("D_x2", D_x2, stepcnt)
                    writer.add_scalar("D_G_z1", D_G_z1, stepcnt)
                    writer.add_scalar("D_G_z2", D_G_z2, stepcnt)
                    writer.add_scalar("D_loss", dloss, stepcnt)
                    writer.add_scalar("G_loss", gloss, stepcnt)

                after = time.time()
                print("{}th iter\tD(x, y): {:.4f}\tD(x, y_fake): {:.4f}\tD(G(z, y)): {:.4f}\tD(G(z, y)): {:.4f}\t{:.4f}s per loop".format(it+1, D_x, D_x2, D_G_z1, D_G_z2, (after-elapsed) / args.log_every), flush=True)

            if it % args.image_every == (args.image_every - 1):
                im = gen(fixed_noise, fixed_label)
                grid = vutils.make_grid(im, normalize=True)
                writer.add_image('epoch {}'.format(ep+1), grid, stepcnt+1)

            stepcnt += 1

        '''log losses per epoch'''
        print("epoch [{}/{}] done | Disc loss: {:.6f} \t Gen loss: {:.6f}".format(ep+1, args.num_epoch, Dloss, Gloss), flush=True)
        if args.use_tensorboard:
            writer.add_text("epoch loss", "epoch [{}/{}] done | Disc loss: {:.6f} \t Gen loss: {:.6f}".format(ep+1, args.num_epoch, Dloss, Gloss), ep+1)


        '''save models'''
        if ep % args.save_model_every == (args.save_model_every - 1):
            torch.save(gen.state_dict(), "../model/gen_epoch_{}.model".format(ep+1))
            torch.save(dis.state_dict(), "../model/dis_epoch_{}.model".format(ep+1))

            '''save and add images based on fixed noise and labels'''
            img = gen(fixed_noise, fixed_label).detach().cpu()
            grid = vutils.make_grid(img, normalize=True)
            if args.use_tensorboard:
                writer.add_image('epoch {}'.format(ep+1), grid, ep+1)
            vutils.save_image(grid, "../out/fixed_labels_ep{}.png".format(ep+1))

            if not os.path.exists('../out/epoch-{}'.format(ep+1)):
                os.mkdir('../out/epoch-{}'.format(ep+1))

            '''save images based on fixed noise and random one-hot attribute labels'''
            for i in range(fsize):
                att = torch.zeros(args.show_size, fsize).to(device)
                att[:, i] = 1
                img = gen(fixed_noise, att).detach().cpu()
                grid = vutils.make_grid(img, normalize=True)
                vutils.save_image(grid, "../out/epoch-{}/{}.png".format(ep+1, attnames[i]))

    print("end training")
    if args.use_tensorboard:
        writer.add_text("Text", "end training")
        writer.close()

def main():

    '''
    optional arguments

    use_tensorboard : True to enable tensorboardX logging
    log_name : name to append on end of log file on Tensorboard
    log_every : log loss and image every ? iterations
    save_model_every : save model of gen and dis every ? epochs
    num_epoch : number of epochs to train
    learning_rate : learning rate of Adam optimizer
    betas : betas of Adam optimizer
    batch_size : batch size for training
    show_size : number of images to show
    root_dir : full path of data. should be parent for images and annotations
    img_dir : relative path of directory containing images
    ann_dir : relative path of file containing annotations of attributes.
    cuda_device : gpu device to use for training
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--image_every', type=int, default=500)
    parser.add_argument('--save_model_every', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--show_size', type=int, default=64)
    parser.add_argument('--root_dir', type=str, default='../../../local/CelebA/')
    parser.add_argument('--img_dir', type=str, default='img_align_celeba')
    parser.add_argument('--ann_dir', type=str, default='list_attr_celeba.csv')
    parser.add_argument('--cuda_device', type=str, default='cuda')
    parser.add_argument('--nz', type=int, default=100)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
