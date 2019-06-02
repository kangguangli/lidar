import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.variable import Variable
from torch.nn.init import xavier_uniform_, zeros_

import math
import numpy as np

#notion while k = 2n + 1, s = 1, let p = n, then we have o = i

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel = 3, stride = 1, downsample = None, groups = 1,
                 base_width = 64, dilation = 1, norm_layer = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel, stride, padding = 1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes, planes, kernel, stride, padding = 1)
        self.bn2 = norm_layer(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size = 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding = 1, stride = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding = 1, stride = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )



class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = BasicBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x



class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride = 2) # o = 2i
        self.conv = BasicBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



def depthToPointCloud(arr):

    result = np.zeros((arr.shape[0] * arr.shape[1], 3))

    width = arr.shape[0]
    height = arr.shape[1]

    theta_w = (35 / 180) * math.pi
    theta_h = (90 / 180) * math.pi
    alpha_w = 2 * math.pi - theta_w / 2 
    alpha_h = (math.pi - theta_h) / 2

    for i in range(width):
        for j in range(height):

            gamma_h = alpha_h + j * (theta_h / height)
            x = arr[i][j] / math.tan(gamma_h)

            gamma_w = alpha_w + i * (theta_w / width)
            y = arr[i][j] * math.tan(gamma_w)

            result[i * height + j] = np.array([arr[i][j], x, -y])
    return result




class DepthToPointCloudFucntion(Function):

    def forward(self, input):

        self.size = input.size()

        output = depthToPointCloud(input.detach().cpu().numpy())
        output = Variable(torch.from_numpy(output).float())
        return output
 
    def backward(self, output_grad): 
        input_grad = torch.FloatTensor(self.size)#output_grad.clone()
        input_grad.fill_(torch.mean(output_grad))
        return input_grad




# class DepthToPointCloud(nn.Module):

#     def __init__(self):
#          super(DepthToPointCloud, self).__init__()

def depth2pc(depth):
        #  return DepthToPointCloudFucntion()(input)

        check_sizes(depth, 'depth', 'B1WH')

        batch_size, _, img_width, img_height = depth.size()

        w_range = torch.arange(0, img_width).view(1, img_width, 1).expand(batch_size, 1, img_width, img_height).type_as(depth)
        h_range = torch.arange(0, img_height).view(1, 1, img_height).expand(batch_size, 1, img_width, img_height).type_as(depth)

        theta_w = (35 / 180) * math.pi
        theta_h = (90 / 180) * math.pi
        alpha_w = 2 * math.pi - theta_w / 2 
        alpha_h = (math.pi - theta_h) / 2

        gamma_h = alpha_h + h_range * (theta_h / img_height)
        x = depth / torch.tan(gamma_h)

        gamma_w = alpha_w + w_range * (theta_w / img_width)
        y = depth * torch.tan(gamma_w)

        res = torch.stack([depth, x, -y], dim = 1)
        res = res.reshape(batch_size, 3, -1)
        res = torch.transpose(res, 1, 2)

        return res


def getSIlog(gt, pred):

    # _pred = pred + 1.0
    # _gt = gt + 1.0

    d_i = torch.log(torch.clamp(pred, min = 1e-3)) - torch.log(torch.clamp(gt, min = 1e-3))
    b, c, h, w = d_i.size()
    n = h * w
    _silog = torch.sum(torch.sum(torch.pow(d_i, 2), 3), 2) / n - torch.pow(torch.sum(torch.sum(d_i, 3), 2), 2) / (n * n)
    silog = torch.mean(_silog)

    return silog


class UNetR(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNetR, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.softplus(x)#F.relu(x)#x#F.sigmoid(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)  



def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))