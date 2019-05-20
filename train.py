import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd.variable import Variable

from tensorboardX import SummaryWriter

import numpy as np

import argparse
import time
import os

from chamfer_distance import ChamferLoss

from logger import Logger
import data_loader
import models

parser = argparse.ArgumentParser(description = 'Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-j', '--workers', default = 1, type = int, metavar = 'N',
                    help = 'number of data loading workers')

parser.add_argument('--epochs', default = 1, type = int, metavar = 'N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default = 4, type = int,
                    metavar = 'N', help = 'batch size')

parser.add_argument('--lr', '--learning-rate', default = 2e-4, type = float,
                    metavar = 'LR', help='initial learning rate')

parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                    help = 'momentum for sgd, alpha parameter for adam')

parser.add_argument('--beta', default = 0.999, type = float, metavar = 'M',
                    help = 'beta parameters for adam')

parser.add_argument('--weight-decay', '--wd', default = 0, type = float,
                    metavar = 'W', help = 'weight decay')

parser.add_argument('--seed', default = 0, type = int, help = 'seed for random functions, and network initialization')

best_error = -1
n_iter = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def main():

    global best_error, n_iter, device, use_cuda
    args = parser.parse_args()
    args.save_path = 'runs/'
 
    use_cuda = False

    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print("Loading Data")
    train_set = data_loader.DepthData('/home/data_kitti/formatted', '/home/data_kitti/raw')
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers, pin_memory = True)

    val_set = data_loader.DepthData('/home/data_kitti/formatted', '/home/data_kitti/raw', train = False)
    val_loader = torch.utils.data.DataLoader(val_set,
        batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers, pin_memory = True)

    print("Load Train Data {}".format(train_set.__len__()))
    print("Load Val Data {}".format(val_set.__len__()))

    print('Creating Model')
    model = models.UNetR(1, 1).to(device)
    model.init_weights()

    cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

    print('Setting Adam Slover')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay)

    training_writer = SummaryWriter(args.save_path)
    
    for epoch in range(args.epochs):

        train_loss = train(args, train_loader, model, optimizer, 0, None, None)

        val_loss = validate(args, val_loader, model, optimizer, 0, None)

        if best_error < 0:
            best_error = val_loss
            save(model, os.path.join(args.save_pth, 'best'))
        elif val_loss < best_error:
            best_error = val_loss
            save(model, os.path.join(args.save_pth, 'best'))

        print("{}_th with loss {}".format(n_iter, val_loss))

    print('Finished')



def train(args, train_loader, model, optimizer, epoch_size, logger, train_writer):

    global n_iter, device

    losses = []
    model.train()


    loss_layer = ChamferLoss()

    for i, (img, depth, _) in enumerate(train_loader):

        output = model(img)

        output_points = models.depth2pc(output)
        gt_points = models.depth2pc(depth)

        loss = loss_layer(output_points, gt_points)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())

    n_iter += 1
    
    return torch.mean(torch.Tensor(losses))

    
@torch.no_grad()
def validate(args, val_loader, model, optimizer, epoch_size, logger):

    global n_iter, device

    losses = []

    model.eval()

    loss_layer = ChamferLoss()

    for i, (img, depth, _) in enumerate(val_loader):

        output = model(img)

        output_points = models.depth2pc(output)
        gt_points = models.depth2pc(depth)

        loss = loss_layer(output_points, gt_points)

        losses.append(loss.detach().cpu().numpy())
    
    return torch.mean(torch.Tensor(losses))



def save(net, path):

    torch.save(net.state_dict(), path)



if __name__ == '__main__':
    main()