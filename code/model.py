import torch
import torch.nn as nn
import torch.nn.utils as utils

from torchvision.models import resnet50, resnet152

class Resblock(nn.Module):
    '''
    Residual Block.
    conv-normalization-activation-conv-normalization-activation
    '''
    def __init__(self, in_c, norm_method='Instancenorm', act='LeakyReLU'):
        super(Resblock, self).__init__()
        self.model = nn.Sequential(
                 Conv(in_c, in_c, ksize=3, stride=1),
                 Normalization(in_c, method=norm_method),
                 Activation(option=act),
                 Conv(in_c, in_c, ksize=3, stride=1),
                 Normalization(in_c, method=norm_method),
                 Activation(option=act)
                 )
    def forward(self, x):
        return x + self.model(x)

class Down(nn.Module):
    '''
    Downsampling Layer
    conv-normalization-activation
    (bs x C x H x W) -> (bs x 2C x H/2 x W/2)
    '''
    def __init__(self, in_c, norm_method='Instancenorm', act='LeakyReLU'):
        super(Down, self).__init__()
        self.model = nn.Sequential(
                 Conv(in_c, in_c*2),
                 Normalization(in_c*2, method=norm_method),
                 Activation(option=act),
                 )
    def forward(self, x):
        return self.model(x)

class Up(nn.Module):
    '''
    Upsampling Layer
    conv-normalization-activation
    (bs x C x H x W) -> (bs x C//2 x 2H x 2W)
    '''
    def __init__(self, in_c, norm_method='Instancenorm', act='LeakyReLU'):
        super(Up, self).__init__()
        self.model = nn.Sequential(
                 Deconv(in_c, in_c//2),
                 Normalization(in_c//2, method=norm_method),
                 Activation(option=act),
                 )
    def forward(self, x):
        return self.model(x)



class AttrEncoder(nn.Module):
    '''
    Attribute predictor class (encoder)
    '''
    def __init__(self, outdims=40):
        super(AttrEncoder, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.reslayers = list(self.resnet.children())[:-2]
        self.reslayers.append(nn.Conv2d(2048, 2048, 2))
        self.model = nn.Sequential(*self.reslayers)
        self.affine = nn.Linear(2048, outdims)
        self.act = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        resout = self.model(x)
        resout = resout.view(resout.size(0), -1)
        out = self.act(self.affine(resout))
        return out

class Encoder(nn.Module):
    '''
    Encoder for attributes(y) or identity(z)
    inputs:
        x or x' (bs x 3 x 64 x 64)
    outputs:
        y (bs x ny)
        or
        z (bs x nz)
    '''
    def __init__(self, ftnum, for_y=True):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
                Conv(3, 32, ksize=5, padding=2),
                Normalization(32),
                Activation(),
                Conv(32, 64, ksize=5, padding=2),
                Normalization(64),
                Activation(),
                Conv(64, 128, ksize=5, padding=2),
                Normalization(128),
                Activation(),
                Conv(128, 256, ksize=5, padding=2),
                Normalization(256),
                Activation(),
                )
        if for_y:
            self.linear = nn.Sequential(
                    Dense(256*4*4, 512),
                    Normalization(512, dim=1),
                    Activation(),
                    Dense(512, ftnum),
                    Activation(option='Sigmoid')
                    )

        else:
            self.linear = nn.Sequential(
                    Dense(256*4*4, 4096),
                    Normalization(4096, dim=1),
                    Activation(),
                    Dense(4096, ftnum),
                    Activation(option='Sigmoid')
                    )

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out2 = self.linear(out)
        return out2


class Generator(nn.Module):
    '''
    Generator for cGAN
    inputs:
        z (bs x nz)
        y' (bs x ny)
        in_c = nz+ny
    outputs:
        x' (bs x 3 x 64 x 64)
    '''
    def __init__(self, in_c):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                           Deconv(in_c, 512, padding=0),
                           Normalization(512),
                           Activation(),
                           Deconv(512, 256),
                           Normalization(256),
                           Activation(),
                           Deconv(256, 128),
                           Normalization(128),
                           Activation(),
                           Deconv(128, 64),
                           Normalization(64),
                           Activation(),
                           Deconv(64, 3),
                           Activation(option='Tanh')
                           )
    def forward(self, z, y):
        bs = z.size(0)
        inp = torch.cat((z, y), 1).view(bs, -1, 1, 1)
        out = self.model(inp)
        return out

class Generator_Res(nn.Module):
    '''
    Generator for cGAN
    inputs:
        z (bs x nz)
        y' (bs x ny)
        in_c = nz+ny
    outputs:
        x' (bs x 3 x 224 x 224)
    '''
    def __init__(self, in_c):
        super(Generator_Res, self).__init__()
        self.model = nn.ModuleList([
                           nn.ConvTranspose2d(in_c, 512, kernel_size=7),
                           Resblock(512),
                           Up(512),
                           Resblock(256),
                           Up(256),
                           Resblock(128),
                           Up(128),
                           Resblock(64),
                           Up(64),
                           Resblock(32),
                           Deconv(32, 3),
                           Activation(option='Tanh')
        ])
    def forward(self, x, y):
        bs = x.size(0)
        out = torch.cat((x, y), 1).view(bs, -1, 1, 1)
        for layer in self.model:
            out = layer(out)
        return out

class Discriminator(nn.Module):
    '''
    Discriminator for cGAN
    inputs:
        x or x' (bs x 3 x 64 x 64)
        y (bs x ny)
    outputs:
        out (bs)
    '''
    def __init__(self, ny):
        super(Discriminator, self).__init__()
        self.before = nn.Sequential(
                          Conv(3, 64),
                          Activation(option='LeakyReLU')
                          )
        self.model = nn.Sequential(
                          Conv(64+ny, 128),
                          Normalization(128),
                          Activation(option='LeakyReLU'),
                          Conv(128, 256),
                          Normalization(256),
                          Activation(option='LeakyReLU'),
                          Conv(256, 512),
                          Normalization(512),
                          Activation(option='LeakyReLU'),
                          Conv(512, 1, stride=1, padding=0),
                          Activation(option='Sigmoid')
                          )
    def forward(self, x, y):
        bs = x.size(0)
        ny = y.size(1)
        if y.dim() == 2:
            y = y.unsqueeze(-1).unsqueeze(-1)
        # out1 (bs, 64, 32, 32)
        out1 = self.before(x)
        # y_broadcast (bs, ny, 32, 32)
        y_broadcast = y.expand(bs, ny, 32, 32)
        out_cat = torch.cat((out1, y_broadcast), 1)
        out2 = self.model(out_cat)
        return out2.view(-1)

class Discriminator_Aux(nn.Module):
    '''
    Discriminator for cGAN
    inputs:
        x or x' (bs x 3 x 64 x 64)
    outputs:
        out (bs x 1)
        y (bs x ny)
    '''
    def __init__(self, ny):
        super(Discriminator_Aux, self).__init__()
        self.ny = ny
        self.before = nn.Sequential(
                          Conv(3, 64),
                          Activation(option='LeakyReLU')
                          )
        self.model = nn.Sequential(
                          Conv(64, 128),
                          Normalization(128),
                          Activation(option='LeakyReLU'),
                          Conv(128, 256),
                          Normalization(256),
                          Activation(option='LeakyReLU'),
                          Conv(256, 512),
                          Normalization(512),
                          Activation(option='LeakyReLU'),
                          Conv(512, 1+ny, stride=1, padding=0),
                          Activation(option='Sigmoid')
                          )
    def forward(self, x):
        bs = x.size(0)
        # out1 (bs, 64, 32, 32)
        out1 = self.before(x)
        out2 = self.model(out1)
        out2 = out2.view(bs, 1+self.ny)
        out, y = torch.split(out2, [1, self.ny], dim=1)
        return out.view(-1), y

class Discriminator_Res(nn.Module):
    '''
    Residual Discriminator for cGAN
    inputs:
        x' (bs x 3 x 224 x 224)
        y (bs x ny)
    outputs:
        out (bs)
    '''
    def __init__(self, ny):
        super(Discriminator_Res, self).__init__()
        self.model = nn.ModuleList([
                           Conv(3+ny, 64),
                           Resblock(64),
                           Down(64),
                           Resblock(128),
                           Down(128),
                           Resblock(256),
                           Down(256),
                           Resblock(512),
                           Down(512),
                           Resblock(1024),
                           Conv(1024, 1, ksize=7, padding=0),
                           Activation(option='Sigmoid')
        ])
    def forward(self, x, y):
        bs = x.size(0)
        ny = y.size(1)
        # x (bs, 3, 224, 224)
        if y.dim() == 2:
            y = y.unsqueeze(-1).unsqueeze(-1)
        # y_broadcast (bs, ny, 224, 224)
        y_broadcast = y.expand(bs, ny, 224, 224)
        out = torch.cat((x, y_broadcast), 1)
        for layer in self.model:
            out = layer(out)
        return out.view(-1)





def Deconv(in_c, out_c, padding=1):
    return nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=padding)

def Conv(in_c, out_c, ksize=4, stride=2, padding=1):
    return nn.Conv2d(in_c, out_c, ksize, stride=stride, padding=padding)

def Normalization(C, dim=2, method='Batchnorm'):
    if method == 'Batchnorm':
        if dim == 2:
            return nn.BatchNorm2d(C)
        elif dim == 1:
            return nn.BatchNorm1d(C)
    elif method == 'Instancenorm':
        return nn.InstanceNorm2d(C)
    else:
        raise NotImplementedError()

def Activation(option='ReLU'):
    if option == 'ReLU':
        return nn.ReLU()
    elif option == 'LeakyReLU':
        return nn.LeakyReLU()
    elif option == 'Tanh':
        return nn.Tanh()
    elif option == 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError()

def Dense(in_n, out_n):
    return nn.Linear(in_n, out_n)

def init_weights(m, std=0.01):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=std)
        m.bias.data.fill_(0)
