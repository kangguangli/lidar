import numpy as np
import os
import math
from PIL import Image

image_size = [1200, 370]
image_size_sfm = [416, 128] 
velo_num = 85504


def getSingle(velo_file: str, root = 'training', img_dir = 'image_2', velo_dir = 'velodyne'):

    img = Image.open(os.path.join(root, img_dir, velo_file.replace('.bin', '.png')))
    img = img.resize(image_size)
    img = img.convert('L')
    img = np.array(img)

    labels = np.fromfile(os.path.join(root, velo_dir, velo_file).replace('.png', '.bin'), dtype = np.float32)
    labels = labels.reshape((-1, 4))
    np.random.shuffle(labels)

    return img, labels[:velo_num, :3]



def getData(root = 'training', img_dir = 'image_2', velo_dir = 'velodyne'):

    for i in os.listdir(os.path.join(root, velo_dir)):
        yield getSingle(i, root = root, img_dir = img_dir, velo_dir = velo_dir)



def getBatchData(root = 'training', img_dir = 'image_2', velo_dir = 'velodyne', batch_size = 64):

    files = np.array((os.listdir(os.path.join(root, velo_dir))))
    files = np.array_split(files, files.shape[0] / batch_size)

    for fs in files:

        n = len(fs)

        imgs = np.ones((n, 1, 600, 1000)) #temp
        labels = np.ones((n, 1, velo_num, 3))

        i = 0

        for f in fs:
            img, label = getSingle(f, root = root, img_dir = img_dir, velo_dir = velo_dir)
            imgs[i, 0, :, :] = img
            labels[i, 0, :, :] = label
            i += 1

        yield imgs, labels


def getImageData(file):

    img = Image.open(file)
    img = img.resize(image_size)
    img = img.convert('L')
    img = np.array(img)

    return np.transpose(img)


def getSingleDepth(file, img_dir, depth_dir):

    img = Image.open(os.path.join(img_dir, file))
    img = img.resize(image_size)
    img = img.convert('L')
    img = np.array(img)

    depth = Image.open(os.path.join(depth_dir, file))
    depth = depth.resize(image_size)
    # depth = depth.convert('L')
    depth = np.array(depth)


    return img, depth



def getBatchDepth_Old(img_dir, depth_dir, batch_size = 64):

    files = np.array(os.listdir(depth_dir))
    files = np.array_split(files, int(files.shape[0] / batch_size))

    yield len(files)

    for fs in files:

        n = len(fs)

        imgs = np.ones((n, 1, image_size[0], image_size[1])) #temp
        depths = np.ones((n, 1, image_size[0], image_size[1]))

        i = 0

        for f in fs:
            img, depth = getSingleDepth(f, img_dir, depth_dir)
            imgs[i, 0, :, :] = np.transpose(img)
            depths[i, 0, :, :] = np.transpose(depth)
            i += 1

        yield imgs, depths


def getBatchDepthS(root, image_channel = 'image_02', batch_size = 64):


    dirs = os.listdir(root)

    total = []

    for dir in dirs:

        if not os.path.isdir(os.path.join(root, dir)):
            continue

        _img_dir = os.path.join(root, dir, image_channel, 'data')
        _depth_dir = os.path.join(root, dir, 'groundtruth', image_channel)

        total.append(list(map(lambda x : (x, _img_dir, _depth_dir), os.listdir(_depth_dir))))

    total = np.array(total)
    piles = np.array_split(total, int(total.shape[0] / batch_size))

    yield len(piles)    

    for p in piles:

        n = len(p)

        imgs = np.ones((n, 1, image_size[0], image_size[1])) 
        depths = np.ones((n, 1, image_size[0], image_size[1]))

        i = 0

        for f, _img_dir, _depth_dir in p:
            img, depth = getSingleDepth(f, _img_dir, _depth_dir)
            imgs[i, 0, :, :] = np.transpose(img)
            depths[i, 0, :, :] = np.transpose(depth)
            i += 1

        yield imgs, depths



def getSingleDepthSfmLearner(img_file, depth_file):

    img = Image.open(img_file)
    # if img.size != image_size_sfm:
    #     img = img.resize(image_size)
    img = img.convert('L')
    img = np.array(img)

    depth = np.load(depth_file)
    # depth = depth.resize(image_size)
    # depth = depth.convert('L')
    # depth = np.array(depth)

    return img, depth


def getBatchDataFromSfmLearner(root, train = True, batch_size = 64):

    scene_list_path = os.path.join(root, 'train.txt') if train else os.path.join(root, 'val.txt')
    scene_list = [os.path.join(root, folder[:-1]) for folder in open(scene_list_path)]

    total = []
    for scene in scene_list:

        if not os.path.isdir(scene):
            continue

        imgs = [os.path.join(scene, f) for f in os.listdir(scene) if f.find('jpg') != -1]
        total.extend(list(map(lambda x : (x, x.replace('jpg', 'npy')), imgs)))

    total = np.array(total)
    piles = np.array_split(total, math.ceil(total.shape[0] / batch_size))

    yield len(piles)   

    for p in piles:

        n = len(p)

        imgs = np.ones((n, 1, image_size_sfm[0], image_size_sfm[1])) 
        depths = np.ones((n, 1, image_size_sfm[0], image_size_sfm[1]))

        i = 0

        for _img, _depth in p:
            img, depth = getSingleDepthSfmLearner(_img, _depth)
            imgs[i, 0, :, :] = np.transpose(img)
            depths[i, 0, :, :] = np.transpose(depth)
            i += 1

        yield imgs, depths


