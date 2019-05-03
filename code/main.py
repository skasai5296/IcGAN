import argparse
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset import DeepFashion, CelebA, SUN_Attributes, collate_fn, randomsample
from model import Generator, Discriminator, init_weights, Generator_Res, Discriminator_Res

def calc_gradient_penalty(netD, LAMBDA, real_data, fake_data, y):
    #print(real_data.size())
    bs = real_data.size(0)
    alpha = torch.rand(bs, 1)
    alpha = alpha.unsqueeze(-1).unsqueeze(-1)
    #print(alpha.size())
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda() if torch.cuda.is_available else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if torch.cuda.is_available():
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, y)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def train(args):

    print(args)

    if args.use_tensorboard:
        log_name = "{}_im{}_lr{}_bs{}_nz{}_res{}_wasser{}".format(args.dataset, args.image_size, args.learning_rate, args.batch_size, args.nz, args.residual, args.wgan)
        print("logging to runs/{}".format(log_name))
        writer = SummaryWriter(log_dir=os.path.join('runs', log_name))

    device = torch.device('cuda' if torch.cuda.is_available() and args.enable_cuda else 'cpu')

    # transforms applied
    transform = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # dataset and dataloader for training
    if args.dataset == 'celeba':
        img_dir = 'img_align_celeba'
        ann_dir = 'list_attr_celeba.csv'
        train_dataset = CelebA(args.root_dir, img_dir, ann_dir, transform=transform)
        attnames = list(train_dataset.df.columns)[1:]
    elif args.dataset == 'sunattributes':
        img_dir = 'images'
        ann_dir = 'SUNAttributeDB'
        train_dataset = SUN_Attributes(args.root_dir, img_dir, ann_dir, transform=transform)
        attnames = train_dataset.attrnames
    else:
        raise NotImplementedError()

    fsize = train_dataset.feature_size

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=8)

    '''
    dataloader returns dictionaries.
    sample : {'image' : (bs, H, W, 3), 'attributes' : (bs, fsize)}
    '''


    # model, optimizer, criterion
    if args.residual:
        gen = Generator_Res(in_c = args.nz + fsize)
        dis = Discriminator_Res(ny=fsize)
    else:
        gen = Generator(in_c = args.nz + fsize)
        dis = Discriminator(ny=fsize)
    gen_params = 0
    dis_params = 0
    for params in list(gen.parameters())[::2]:
        gen_params += params.numel()
    for params in list(dis.parameters())[::2]:
        dis_params += params.numel()
    print("# of parameters in generator : {}".format(gen_params))
    print("# of parameters in discriminator : {}".format(dis_params))
    gen = gen.to(device)
    dis = dis.to(device)

    # initialize weights for conv layers
    gen.apply(init_weights)
    dis.apply(init_weights)

    if args.opt_method == 'Adam':
        gen_optim = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=args.betas)
        dis_optim = optim.Adam(dis.parameters(), lr=args.learning_rate, betas=args.betas)
    if args.opt_method == 'SGD':
        gen_optim = optim.SGD(gen.parameters(), lr=args.learning_rate, momentum=args.momentum)
        dis_optim = optim.SGD(dis.parameters(), lr=args.learning_rate, momentum=args.momentum)
        gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optim, patience=args.patience)
        dis_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dis_optim, patience=args.patience)
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.show_size, args.nz).to(device)
    # fixed_label = torch.zeros(args.show_size, fsize).to(device)
    fixed_label = randomsample(train_dataset, args.show_size).to(device)

    label = torch.empty(args.batch_size).to(device)
    noise = torch.empty(args.batch_size, args.nz).to(device)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if args.use_tensorboard:
        writer.add_text("Text", "begin training, lr={}".format(args.learning_rate))
    print("begin training, lr={}".format(args.learning_rate), flush=True)
    stepcnt = 0
    gen.train()
    dis.train()

    real_tar = 0.9
    fake_tar = 0.1

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
            # print(out.size(), real_label.size())
            if not args.wgan:
                real_dis_loss = criterion(out, real_label)
            else:
                real_dis_loss = out.mean()
            real_dis_loss.backward()
            D_x = out.mean().detach()

            # train on fake images and real attributes, target are zeros (fake)
            fake_label = torch.full_like(label, fake_tar)
            z = torch.randn_like(noise)
            y_real = y.clone().detach()
            x_fake = gen(z, y_real)
            out = dis(x_fake, y_real)
            if not args.wgan:
                fake_dis_loss = criterion(out, fake_label)
            else:
                fake_dis_loss = out.mean()
            fake_dis_loss.backward()
            D_G_z1 = out.mean().detach()

            # train using gradient penalty
            if args.wgan:
                gp = calc_gradient_penalty(dis, 0.1, x_real, x_fake, y_real)
                gp.backward()

            dis_optim.step()
            errD = (real_dis_loss + fake_dis_loss) / 2

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
                    writer.add_scalar("D_G_z1", D_G_z1, stepcnt)
                    writer.add_scalar("D_G_z2", D_G_z2, stepcnt)
                    writer.add_scalar("D_loss", dloss, stepcnt)
                    writer.add_scalar("G_loss", gloss, stepcnt)

                after = time.time()
                print("{}th iter\tD(x, y): {:.4f}\tD(G(z, y))_1: {:.4f}\tD(G(z, y))_2: {:.4f}\t{:.4f}s per loop".format(it+1, D_x, D_G_z1, D_G_z2, (after-elapsed) / args.log_every), flush=True)

            if it % args.image_every == (args.image_every - 1):
                im = gen(fixed_noise, fixed_label)
                grid = vutils.make_grid(im, normalize=True)

                if args.use_tensorboard:
                    writer.add_image('iter {}'.format(it+1), grid, stepcnt+1)
                    print("logged image, iter {}".format(it+1))

            stepcnt += 1

        # learning rate scheduler
        if args.opt_method is 'SGD':
            gen_scheduler.step(g_loss)
            dis_scheduler.step(d_loss)

        '''log losses per epoch'''
        print("epoch [{}/{}] done | Disc loss: {:.6f} \t Gen loss: {:.6f}".format(ep+1, args.num_epoch, Dloss, Gloss), flush=True)
        if args.use_tensorboard:
            writer.add_text("epoch loss", "epoch [{}/{}] done | Disc loss: {:.6f} \t Gen loss: {:.6f}".format(ep+1, args.num_epoch, Dloss, Gloss), ep+1)


        '''save models and generated images on fixed labels'''
        if ep % args.save_model_every == (args.save_model_every - 1):
            savedir = os.path.join(args.model_path, "{}/res_{}".format(args.dataset, args.residual))
            if not os.path.exists(savedir):
                os.makedirs(savedir, exist_ok=True)
            gen_pth = "gen_epoch_{}.ckpt".format(ep+1)
            dis_pth = "dis_epoch_{}.ckpt".format(ep+1)

            torch.save(gen.state_dict(), os.path.join(savedir, gen_pth))
            torch.save(dis.state_dict(), os.path.join(savedir, dis_pth))
            print("saved model to {}".format(savedir))

            '''save and add images based on fixed noise and labels'''
            imdir = os.path.join(args.output_path, "{}/res_{}/cgan_epoch_{}".format(args.dataset, args.residual, ep+1))
            if not os.path.exists(imdir):
                os.makedirs(imdir, exist_ok=True)
            img = gen(fixed_noise, fixed_label).detach().cpu()
            grid = vutils.make_grid(img, normalize=True)
            if args.use_tensorboard:
                writer.add_image('epoch {}'.format(ep+1), grid, ep+1)
            vutils.save_image(grid, os.path.join(imdir, "original.png"))

            '''save images based on fixed noise and labels, make attribute 1'''
            for i in range(fsize):
                att = fixed_label.new_tensor(fixed_label.data)
                att[:, i] = 1
                img = gen(fixed_noise, att).detach().cpu()
                grid = vutils.make_grid(img, normalize=True)
                vutils.save_image(grid, os.path.join(imdir, "{}.png".format(attnames[i].replace("/", ""))))
            print("saved original and transformed images in {}".format(imdir), flush=True)

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
    wgan : 'none' for no gp, 'wgangp' for improved wasserstein gan
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--image_every', type=int, default=500)
    parser.add_argument('--save_model_every', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--opt_method', choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--patience', type=float, default=5)
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--show_size', type=int, default=64)
    parser.add_argument('--dataset', choices=['celeba', 'sunattributes'], default='celeba')
    parser.add_argument('--root_dir', type=str, default='../../dsets/CelebA/')
    parser.add_argument('--enable_cuda', action='store_true')
    parser.add_argument('--model_path', type=str, default='../model')
    parser.add_argument('--output_path', type=str, default='../out')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--wgan', action='store_true')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
