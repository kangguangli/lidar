import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.autograd.variable import Variable

from logger import Logger
import data_loader

nc = 1
ndf = 8
ngf = 1
ngpu = 1


def fake_loader():

    for i in range(2):
        yield np.random.rand(1, 1, 1200, 370), np.random.rand(1, 1, 1200, 370)


class Discriminator(nn.Module):

    def __init__(self, ngpu):

        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is nc * 1200 * 370

            nn.Conv2d(nc, ndf, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 600 x 185

            nn.Conv2d(ndf, ndf * 2, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 300 x 92

            nn.Conv2d(ndf * 2, ndf * 4, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 150 x 46

            nn.Conv2d(ndf * 4, ndf * 8, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 75 x 23

            nn.Conv2d(ndf * 8, ndf * 4, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 37 x 11

            nn.Conv2d(ndf * 4, ndf * 2, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 13 x 5

            nn.Conv2d(ndf * 2, ndf, (4, 4), stride = (3, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 4 x 2

            nn.Conv2d(ndf, 1, (4, 2), stride = (4, 1), padding = (0, 0), bias = False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    
    def forward(self, input):

        return self.main(input)


class Generator(nn.Module):

    def __init__(self, ngpu):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 1 * 1200 * 3700

            nn.Conv2d(1, ngf, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(ngf, ngf * 2, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(ngf * 2, ngf * 4, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace = True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, (4, 4), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

    
    def forward(self, input):

        return self.main(input)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netD = Discriminator(ngpu).to(device)
netG_path = 'runs/models/netG'
netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

d_optimizer = optim.Adam(netD.parameters(), lr=0.0002)
g_optimizer = optim.Adam(netG.parameters(), lr=0.0002)

loss = nn.BCELoss()


def train_discriminator(optimizer, real_data, fake_data):

    n = real_data.size(0)

    optimizer.zero_grad()

    prediction_real = netD(real_data).view(-1)
    label_real = torch.full((n,), 1, device = device) # device = device
    error_real = loss(prediction_real, label_real)
    error_real.backward()

    prediction_fake = netD(fake_data).view(-1)
    label_fake = torch.full((n,), 0, device = device) # device = device
    error_fake = loss(prediction_fake, label_fake)
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake



def train_generator(optimizer, fake_data):
    
    n = fake_data.size(0)

    optimizer.zero_grad()

    prediction = netD(fake_data).view(-1)
    label = torch.full((n,), 1, device = device) # device = device
    error = loss(prediction, label)
    error.backward()

    optimizer.step()

    return error


def save(net, path):

    torch.save(net.state_dict(), path)

def load_genetator(path):

    model = Generator(ngpu)
    model.load_state_dict(torch.load(path))
    return model


def train():

    num_epoches = 2

    logger = Logger(model_name='Depth', data_name='KITTI_DEPTH')

    fixed_input = 'data/depth/2011_09_26_drive_0001_sync/image_02/data/0000000005.png'
    fixed_data = data_loader.getImageData(fixed_input)
    fixed_input = np.ones((1,1,1200,370))
    fixed_input[0, 0, :, :] = fixed_data
    fixed_input = Variable(torch.from_numpy(fixed_input).float()).to(device)
    

    for epoch in range(num_epoches):

        # data = fake_loader()
        data = data_loader.getBatchDepth(
            'data/depth/2011_09_26_drive_0001_sync/image_02/data', 
            'data/depth/2011_09_26_drive_0001_sync/groundtruth/image_02', 
            1
        )
        num_batches = next(data)

        for n_batch, (img, depth) in enumerate(data):

            n = img.shape[0]
            print(n_batch, ' Batch contains ', n, 'imgs.')

            real_data = Variable(torch.from_numpy(depth).float()).to(device)

            img_data = Variable(torch.from_numpy(img).float()).to(device)
            fake_data = netG(img_data).detach()

            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

            fake_data = netG(img_data)
            g_error = train_generator(g_optimizer, fake_data)

            if n_batch % 10 == 0:
                
                # img = fixed.cpu().data
                # real_depth = real_data.cpu().data
                # fake_depth = fake_data.cpu().data


                # logger.log_images(img, n, epoch, n_batch, num_batches, kind = 'img')
                # logger.log_images(real_depth, n, epoch, n_batch, num_batches, kind = 'rd')

                fake_depth = netG(fixed_input).cpu().data

                logger.log_images(fake_depth, n, epoch, n_batch, num_batches)

                logger.display_status(
                    epoch, num_epoches, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

    save(netG, netG_path)
    