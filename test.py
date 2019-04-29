# input size 1000 x 600

# output size 100000 x 3

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd.variable import Variable
from IPython.display import HTML
from PIL import Image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

seed = 999
random.seed(seed)
torch.manual_seed(seed)

data_root = 'data'
image_size = [1000, 600]

def getData():

    for i in os.listdir(os.path.join(data_root, 'img')):
        img = Image.open(os.path.join(data_root, 'img', i))
        img = img.resize(image_size)
        img = img.convert('L')
        img = np.array(img)

        labels = np.fromfile(os.path.join(data_root, 'velo', i).replace('.png', '.bin'), dtype = np.float32)
        labels = labels.reshape((-1, 4))

        yield np.transpose(img), labels[:, :3]


workers = 2
batch_size = 1

nc = 1
nz = 1
ngf = 1
num_epochs = 5
ndf = 1

lr = 0.02
ngpu = 1
beta1 = 0.5

# data_generator = next(getData)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



class Generator(nn.Module):


    def __init__(self, ngpu):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, 1 * 600 * 1000

            nn.ConvTranspose2d(nz, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 2002

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 4006

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 8014

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 16030

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 32062

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 64126

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 128254

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 256510 

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 513022

            nn.ConvTranspose2d(ngf, ngf, (1, 4), stride = (1, 2), padding = (0, 0), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 600 x 1026046

            nn.Conv2d(ngf, ndf, (3, 3), stride = (3, 3), padding = (0, 0), bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 200 x 342015

            nn.Conv2d(ndf, ndf, (3, 3), stride = (3, 4), padding = (0, 0), bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 22 x 85504

            nn.Conv2d(ndf, ndf, (3, 1), stride = (3, 1), padding = (0, 0), bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 7 x 85504

            nn.Conv2d(ndf, 1, (5, 1), stride = (1, 1), padding = (0, 0), bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. 1 x 3 x 85504
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 1 * 3 * 85504

            nn.Conv2d(nc, ndf, (3, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 1 x 28501

            nn.Conv2d(ndf, ndf * 2, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 1 x 9501

            nn.Conv2d(ndf * 2, ndf * 4, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*4) x 1 x 3167


            nn.Conv2d(ndf * 4, ndf * 8, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*8) x 1 x 1056

            nn.Conv2d(ndf * 8, ndf * 16, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace = True),
            # state size.  (ndf*16) x 1 x 352

            nn.Conv2d(ndf * 16, ndf * 32, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*32) x 1 x 117

            nn.Conv2d(ndf * 32, ndf * 16, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*16) x 1 x 39

            nn.Conv2d(ndf * 16, ndf * 8, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*8) x 1 x 13

            nn.Conv2d(ndf * 8, ndf * 4, (1, 4), stride = (1, 3), padding = (0, 1), bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*4) x 1 x 4

            nn.Conv2d(ndf * 4, 1, (1, 4), stride = (1, 1), padding = (0, 0), bias = False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



loss = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

netD = Discriminator(ngpu)
netG = Generator(ngpu)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


def train_generator(optimizer, real_data : np.ndarray, fake_data : np.ndarray):

    size = real_data.shape[0]

    optimizer.zero_grad()

    #Train on Real Data
    prediction_real = netD(real_data)
    error_real = loss(prediction_real, Variable(torch.ones(size, 1)))
    error_real.backward()

    prediction_fake = netD(fake_data)

    



img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

# for epoch in range(num_epochs):
#     for data in enumerate(getData):

#         netD.zero_grad()
#         real_cpu = data[0].to(device)
#         b_size = real_cpu.size(0)
