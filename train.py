import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd.variable import Variable

from tensorboardX import SummaryWriter

import numpy as np
from PIL import Image

import argparse
import time
import os

from chamfer_distance import ChamferLoss

from logger import Logger
import data_loader
import models

parser = argparse.ArgumentParser(description = 'Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-j', '--workers', default = 8, type = int, metavar = 'N',
                    help = 'number of data loading workers')

parser.add_argument('--epochs', default = 100, type = int, metavar = 'N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default = 16, type = int,
                    metavar = 'N', help = 'batch size')

parser.add_argument('--lr', '--learning-rate', default = 2e-4, type = float,
                    metavar = 'LR', help='initial learning rate')

parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                    help = 'momentum for sgd, alpha parameter for adam')

parser.add_argument('--beta', default = 0.999, type = float, metavar = 'M',
                    help = 'beta parameters for adam')

parser.add_argument('--weight-decay', '--wd', default = 0, type = float,
                    metavar = 'W', help = 'weight decay')

parser.add_argument('--channel', '--c', default = 3, type = int,
                    metavar = 'C', help = 'input channel')

parser.add_argument('--seed', default = 0, type = int, help = 'seed for random functions, and network initialization')

best_error = -1
n_iter = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")



def main():

    global best_error, n_iter, device, use_cuda
    args = parser.parse_args()
    args.save_path = 'runs/'

    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print("Loading Data")
    train_set = data_loader.DepthData('/home/data_kitti/formatted', '/home/data_kitti/raw', args.channel)
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers, pin_memory = True)

    val_set = data_loader.DepthData('/home/data_kitti/formatted', '/home/data_kitti/raw', args.channel, train = False)
    val_loader = torch.utils.data.DataLoader(val_set,
        batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers, pin_memory = True)

    print("Load Train Data {}".format(train_set.__len__()))
    print("Load Val Data {}".format(val_set.__len__()))
    print("Input Data Channel {}".format(args.channel))

    print('Creating Model')
    model = models.UNetR(args.channel, 1).to(device)
    model.init_weights()

    cudnn.benchmark = True
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    print('Setting Adam Slover')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay)

    training_writer = SummaryWriter(args.save_path)
    
    for epoch in range(args.epochs):

        train_loss = train(args, train_loader, model, optimizer, 0, None, None)

        val_loss = validate(args, val_loader, model, optimizer, 0, None)
        # val_loss = train_loss

        save(model, os.path.join(args.save_path, 'pth'))
        if best_error < 0:
            best_error = val_loss
            save(model, os.path.join(args.save_path, 'best'))
        elif val_loss < best_error:
            best_error = val_loss
            save(model, os.path.join(args.save_path, 'best'))

        print("{}_th with train_loss {} val_loss {}".format(n_iter, train_loss, val_loss))

    print('Finished')


def random_crop(pred, gt, target_w = 100, target_h = 10):

    batch_size, _, width, height = pred.size()

    crop_w = np.random.randint(0, width - target_w + 1)
    crop_h = np.random.randint(0, height - target_h + 1)

    crop_w_end = crop_w + target_w
    crop_h_end = crop_h + target_h

    if pred.is_cuda and gt.is_cuda:

        _pred = pred[:, :, crop_w : crop_w_end, crop_h : crop_h_end].cuda()
        _gt = gt[:, :, crop_w : crop_w_end, crop_h : crop_h_end].cuda()

    else:

        _pred = pred[:, :, crop_w : crop_w_end, crop_h : crop_h_end].cpu()
        _gt = gt[:, :, crop_w : crop_w_end, crop_h : crop_h_end].cpu()

    return _pred, _gt



def train(args, train_loader, model, optimizer, epoch_size, logger, train_writer):

    global n_iter, device

    losses = []
    model.train()


    loss_layer = ChamferLoss()

    for i, (img, depth, _) in enumerate(train_loader):

        img = img.to(device)
        depth = depth.to(device)

        output = model(img)

            # _output, _depth = random_crop(output, depth)
            # _output, _depth = output, depth

        output_points = models.depth2pc(output)
        gt_points = models.depth2pc(depth)

        cd_loss = loss_layer(output_points, gt_points)
        
        cd_loss = torch.mean(cd_loss)


        silog = models.getSIlog(depth, output)
        var = 1 / torch.var(output_points) * 0.2
        smooth = torch.nn.functional.smooth_l1_loss(output, depth)


        print('{}_Train Loss: cd loss {}, silog {}, var {}, smooth {}'.format(i, cd_loss, silog, var, smooth))

        loss = cd_loss + silog + var #+ smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())

    n_iter += 1

    return np.mean(np.array(losses))

    
@torch.no_grad()
def validate(args, val_loader, model, optimizer, epoch_size, logger):

    global n_iter, device

    losses = []

    model.eval()

    loss_layer = ChamferLoss()

    for i, (img, depth, _) in enumerate(val_loader):

        img = img.to(device)
        depth = depth.to(device)

        output = model(img)
        
        output_points = models.depth2pc(output)
        gt_points = models.depth2pc(depth)

        cd_loss = loss_layer(output_points, gt_points)
        cd_loss = torch.mean(cd_loss)

        silog = models.getSIlog(depth, output)
        var = 1 / torch.var(output_points) * 0.2
        smooth = torch.nn.functional.smooth_l1_loss(output, depth)

        print('{}_Val Loss: cd loss {}, silog {}, var {}, smooth {}'.format(i, cd_loss, silog, var, smooth))

        loss = cd_loss + silog + var #+ smooth

        losses.append(loss.detach().cpu().numpy())
    
    return np.mean(np.array(losses))



def save(net, path):

    torch.save({'state_dict':net.module.state_dict()}, path)



@torch.no_grad()
def run_interface(img_path, save_path, channels):

    img = Image.open(img_path)
    img = np.array(img)
    img = np.transpose(img, [2, 1, 0])
    img = img[np.newaxis, :]
    img = Variable(torch.from_numpy(img).float()).to(device)

    net = models.UNetR(channels, 1).to(device)
    weights = torch.load(save_path)
    net.load_state_dict(weights['state_dict'])
    net.eval()

    output = net(img)
    output_points = models.depth2pc(output)
    return output_points.detach().cpu().numpy()



if __name__ == '__main__':
    main()