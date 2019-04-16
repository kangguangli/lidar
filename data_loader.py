import numpy as np
import os
from PIL import Image

image_size = [1000, 600]

def getData(root = 'training', img_dir = 'image_2', velo_dir = 'velodyne'):

    for i in os.listdir(os.path.join(root, velo_dir)):

        img = Image.open(os.path.join(root, img_dir, i.replace('.bin', '.png')))
        img = img.resize(image_size)
        img = img.convert('L')
        img = np.array(img)

        labels = np.fromfile(os.path.join(root, velo_dir, i).replace('.png', '.bin'), dtype = np.float32)
        labels = labels.reshape((-1, 4))

        yield img, labels[:, :3]

