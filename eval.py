import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.variable import Variable
from torch.nn.init import xavier_uniform_, zeros_

import math
import numpy as np
import os
from PIL import Image

import models
import data_loader
import DispNetS
from chamfer_distance import ChamferLoss


import scipy.misc


use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def main():

    global device

    val_set = data_loader.DepthData('/home/data_kitti/formatted', '/home/data_kitti/raw', 3, train = False)
    val_loader = torch.utils.data.DataLoader(val_set,
        batch_size = 16, shuffle = True,
        num_workers = 8, pin_memory = True)

    our_net = models.UNetR(3, 1).to(device)
    weights = torch.load('runs/best_b_sp_3_silog_var_smooth')
    our_net.load_state_dict(weights['state_dict'])

    our_loss = eval(our_net, val_loader)

    print('ours: ', our_loss)

    disp_net = DispNetS.DispNetS().to(device)
    weights = torch.load('runs/disp_best')
    disp_net.load_state_dict(weights['state_dict'])
    sfm_loss = eval_sfmlearner(disp_net, val_loader)

    print('sfm: ', sfm_loss)


@torch.no_grad()
def eval(net, data_set):

    global device

    net.eval()

    loss_layer = ChamferLoss()

    losses = np.zeros((data_set.__len__(), 11), dtype = np.float32)

    for i, (img, depth, _) in enumerate(data_set, 0):

        img = img.to(device)
        depth = depth.to(device)

        output = net(img)

        output_points = models.depth2pc(output)
        gt_points = models.depth2pc(depth)

        cd_loss = loss_layer(output_points, gt_points)
        cd_loss = torch.mean(cd_loss)
        losses[i, 0] = cd_loss.detach().cpu().numpy()

        depth = torch.clamp(depth, min = 1e-3)
        output = torch.clamp(output, min = 1e-3)

        silog = models.getSIlog(depth, output)
        losses[i, 1] = silog.detach().cpu().numpy()

        depth_errors = compute_errors(depth.detach().cpu().numpy(), output.detach().cpu().numpy())
        for x in range(9):
            losses[i, 2 + x] = depth_errors[x]

    return np.mean(losses, 0)



@torch.no_grad()
def eval_sfmlearner(net, data_set):

    global device

    net.eval()

    loss_layer = ChamferLoss()

    losses = np.zeros((data_set.__len__(), 11), dtype = np.float32)

    for i, (img, depth, _) in enumerate(data_set, 0):

        img = img.permute(0, 1, 3, 2)
        img = (img/255 - 0.5)/0.2
        assert img.size()[2] == 128

        img = img.to(device)
        depth = depth.to(device)

        output = net(img)
        output = 1 / output
        output = output.permute(0, 1, 3, 2)

        output_points = models.depth2pc(output)
        gt_points = models.depth2pc(depth)

        depth = torch.clamp(depth, min = 1e-3)
        output = torch.clamp(output, min = 1e-3)

        cd_loss = loss_layer(output_points, gt_points)
        cd_loss = torch.mean(cd_loss)
        losses[i, 0] = cd_loss

        silog = models.getSIlog(depth, output)
        losses[i, 1] = silog

        depth_errors = compute_errors(depth.cpu().numpy(), output.cpu().numpy() * np.median(depth.cpu().numpy())/np.median(output.cpu().numpy()))
        for x in range(9):
            losses[i, 2 + x] += depth_errors[x]
    return np.mean(losses, 0)



def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3


@torch.no_grad()
def build(net, img, depth):
    
    global device

    net.eval()

    output = net(img)
    
    output = output.permute(0, 1, 3, 2)
    depth = depth.permute(0, 1, 3, 2)

    output_points = models.depth2pc(output)
    gt_points = models.depth2pc(depth)

    return np.squeeze(output.detach().cpu().numpy()), np.squeeze(output_points.detach().cpu().numpy()), np.squeeze(gt_points.detach().cpu().numpy())


@torch.no_grad()
def build_sfmlearner(net, img, depth):
    
    global device

    net.eval()

    img = img.permute(0, 1, 3, 2)
    img = (img/255 - 0.5)/0.2

    output = net(img)
    output = 1 / output

    # output = output.permute(0, 1, 3, 2)
    depth = depth.permute(0, 1, 3, 2)

    output_points = models.depth2pc(output)
    gt_points = models.depth2pc(depth)

    # return output.detach().cpu().numpy(), output_points.detach().cpu().numpy(), gt_points.detach().cpu().numpy()
    return np.squeeze(output.detach().cpu().numpy()), np.squeeze(output_points.detach().cpu().numpy()), np.squeeze(gt_points.detach().cpu().numpy())



def build_from_files(inputs, outputs):

    global device

    files = os.listdir(inputs)
    files = [f for f in files if f.find('jpg') != -1]

    our_net = models.UNetR(3, 1).to(device)
    weights = torch.load('runs/best')
    our_net.load_state_dict(weights['state_dict'])

    disp_net = DispNetS.DispNetS().to(device)
    weights = torch.load('runs/disp_best')
    disp_net.load_state_dict(weights['state_dict'])

    for f in files:

        img = Image.open(os.path.join(inputs, f))
        img = np.array(img)
        img = np.transpose(img, [2, 1, 0])
        img = img[np.newaxis, :]

        depth = np.load(os.path.join(inputs, f.replace('jpg', 'npy')))
        depth = np.transpose(depth) 
        depth = depth[np.newaxis, np.newaxis, :]

        img = Variable(torch.from_numpy(img).float()).to(device)
        depth = Variable(torch.from_numpy(depth).float()).to(device)

        our_output, our_op, our_gp = build(our_net, img, depth)
        sfm_output, sfm_op, sfm_gp = build_sfmlearner(disp_net, img, depth)

        np.save(os.path.join(outputs, f.replace('jpg', 'our_op')), our_op)
        np.save(os.path.join(outputs, f.replace('jpg', 'our_gp')), our_gp)

        np.save(os.path.join(outputs,f.replace('jpg', 'sfm_op')), sfm_op)
        np.save(os.path.join(outputs,f.replace('jpg', 'sfm_gp')), sfm_gp)

        scipy.misc.toimage(our_output).save(os.path.join(outputs, f.replace('jpg', 'our.jpg')))
        scipy.misc.toimage(sfm_output).save(os.path.join(outputs, f.replace('jpg', 'sfm.jpg')))


if __name__ == '__main__':
    main()