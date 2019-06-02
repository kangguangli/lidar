import numpy as np

from PIL import Image
import math

import pptk


def transform(file):

    img = Image.open(file)
    arr = np.array(img)
    img.close()

    arr = arr

    width = arr.shape[0]
    height = arr.shape[1]

    theta_w = (35 / 180) * math.pi
    theta_h = (90 / 180) * math.pi
    alpha_w = 2 * math.pi - theta_w / 2 
    alpha_h = (math.pi - theta_h) / 2

    res = []

    for i in range(width):
        for j in range(height):

            gamma_h = alpha_h + j * (theta_h / height)
            x = arr[i][j] / math.tan(gamma_h)

            gamma_w = alpha_w + i * (theta_w / width)
            y = arr[i][j] * math.tan(gamma_w)

            
            if arr[i][j] != 0:
                # res.append([arr[i][j],j, -i])
                res.append([arr[i][j], x, -y])

    res = np.array(res)

    pptk.viewer(res)


def transform_array(arr):

    width = arr.shape[0]
    height = arr.shape[1]

    theta_w = (35 / 180) * math.pi
    theta_h = (90 / 180) * math.pi
    alpha_w = 2 * math.pi - theta_w / 2 
    alpha_h = (math.pi - theta_h) / 2

    res = []

    for i in range(width):
        for j in range(height):

            gamma_h = alpha_h + j * (theta_h / height)
            x = arr[i][j] / math.tan(gamma_h)

            gamma_w = alpha_w + i * (theta_w / width)
            y = arr[i][j] * math.tan(gamma_w)

            
            # if arr[i][j] != 0:
                # res.append([arr[i][j],j, -i])
            res.append([arr[i][j], x, -y])

    res = np.array(res)

    # f = pptk.viewer(res)
    return res