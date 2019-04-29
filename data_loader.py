import numpy as np
import os
from PIL import Image

image_size = [1200, 370]
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



def getBatchDepth(img_dir, depth_dir, batch_size = 64):

    files = np.array(os.listdir(depth_dir))
    files = np.array_split(files, int(files.shape[0] / batch_size))

    yield len(files)

    for fs in files:

        n = len(fs)

        imgs = np.ones((n, 1, 1200, 370)) #temp
        depths = np.ones((n, 1, 1200, 370))

        i = 0

        for f in fs:
            img, depth = getSingleDepth(f, img_dir, depth_dir)
            imgs[i, 0, :, :] = np.transpose(img)
            depths[i, 0, :, :] = np.transpose(depth)
            i += 1

        yield imgs, depths