import numpy as np
import os
import math
from PIL import Image

import torch
import torch.utils.data as data
from torch.autograd.variable import Variable


image_size_sfm = [416, 128] 



def getArray(file : str, four = False) -> np.ndarray :

    arr = np.fromfile(file, dtype = 'float32')
    arr = arr.reshape((-1, 4))
    if four != True:
        arr = arr[:, :3]

    return arr



def getSingleDepthSfmLearner(img_file, depth_file, channel = 1):

    img = Image.open(img_file)
    if channel == 1:
        img = img.convert('L')
    img = np.array(img)

    depth = np.load(depth_file)

    return img, depth



def findVelo(f, velo_dir):
    
    _, _, _, _, scene, f = f.split(os.path.sep)
    velo_file = os.path.join(velo_dir, scene[5:10], scene, 'velodyne_points', 'data', f.replace('jpg', 'bin'))
    return velo_file




def getBatchDataFromSfmLearner(root, velo_dir, train = True):

    global image_size_sfm

    scene_list_path = os.path.join(root, 'train.txt') if train else os.path.join(root, 'val.txt')
    scene_list = [os.path.join(root, folder[:-1]) for folder in open(scene_list_path)]

    total = []
    for scene in scene_list:

        if not os.path.isdir(scene):
            continue

        imgs = [os.path.join(scene, f) for f in os.listdir(scene) if f.find('jpg') != -1]
        total.extend(list(map(lambda x : (x, x.replace('jpg', 'npy'), findVelo(x, velo_dir)), imgs)))

    total = np.array(total)
    return total



def angleOfVector(p1, p2):

    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)

    return np.degrees(np.arccos(np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)))



def shouldLeft(point, hfov = 45.0, vfov = 35.0):

    assert (point.shape == (3,))

    h_angle = angleOfVector(np.array([point[0], point[1]]), np.array([1, 0]))
    # v_angle = angleOfVector(np.array([point[0], point[2]]), np.array([0, 1]))

    # print(h_angle, v_angle)

    return h_angle < hfov #and v_angle < vfov



def filterPoint(arr : np.ndarray):

    assert (arr.shape[1] == 3)
    
    sign = np.apply_along_axis(lambda x : shouldLeft(x), 1, arr)
    return arr[sign]


class DepthData(data.Dataset):

    def __init__(self, root, velo_dir, channel, train = True,):

        self.data = getBatchDataFromSfmLearner(root, velo_dir, train)
        self.channel = channel


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        _img, _depth, _velo = self.data[index]
        img, depth = getSingleDepthSfmLearner(_img, _depth, channel = self.channel)
        if self.channel != 1:
            img = np.transpose(img, [2, 1, 0])
        else:
            img = np.transpose(img)
            img = img[np.newaxis, :]
        img = Variable(torch.from_numpy(img).float())

        depth = np.transpose(depth) 
        depth = depth[np.newaxis, :]
        depth = Variable(torch.from_numpy(depth).float())

        return img, depth, _velo

