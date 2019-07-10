import numpy as np

import pptk
import time

import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc

def get_capture_from_pc(arr, specific = True):

    theta, phi, r = (0.04295149, 3.42998147, 12.3728714)

    if specific == True:
        arr = arr[arr[:, 0] > 1.0]

    v = pptk.viewer(arr)
    v.set(show_axis = False)
    v.set(show_info = False)

    if specific == True:
        v.set(theta = theta, phi = phi, r = r)
        v.set(lookat = (15.09189129,  0.4560284 , -0.72079194))

    time.sleep(10)

    v.capture('screenshot.png')
    time.sleep(10)

    # v.close()

    res = None
    with Image.open('screenshot.png') as img:
        img = img.resize((416,128)).convert('RGB')
        res = np.array(img)
    
    return res


def main(inputs, outputs):

    files = os.listdir(inputs)
    files = [f for f in files if f.find('jpg') != -1]

    n = len(files)
    i = 0

    # fig, axs = plt.subplots(n, 7)

    # plt.xticks([])
    # plt.yticks([])

    res = np.zeros((128 * 7, 416 * n, 3))

    for f in files:

        img = Image.open(os.path.join(inputs, f))
        img = np.array(img)
        # img = np.transpose(img, [1, 0, 2]) # 416 * 128 *3

        print(img.shape)

        # axs[i, 0].imshow(img)
        res[0:128, 416 * i: 416 * (i + 1)] = img

        depth = np.load(os.path.join(inputs, f.replace('jpg', 'npy')))
        # depth = np.transpose(depth) # 416 * 128

        print(depth.shape)

        # axs[i, 1].imshow(depth)
        res[128:128 * 2, 416 * i: 416 * (i + 1)] = np.asarray(Image.fromarray(depth).convert('RGB'))

        our_output = Image.open(os.path.join(outputs, f.replace('jpg', 'our.jpg')))
        our_output = np.array(our_output)

        print(our_output.shape)

        # axs[i, 2].imshow(our_output)
        res[128 * 2:128 * 3, 416 * i: 416 * (i + 1)] = np.asarray(Image.fromarray(our_output).convert('RGB'))

        sfm_output = Image.open(os.path.join(outputs, f.replace('jpg', 'sfm.jpg')))
        sfm_output = np.array(sfm_output)

        print(sfm_output.shape)

        # axs[i, 3].imshow(sfm_output)
        res[128 * 3:128 * 4, 416 * i: 416 * (i + 1)] = np.asarray(Image.fromarray(sfm_output).convert('RGB'))

        our_op = np.load(os.path.join(outputs, f.replace('jpg', 'our_op.npy')))
        sfm_op = np.load(os.path.join(outputs,f.replace('jpg', 'sfm_op.npy')))
        gp = np.load(os.path.join(outputs, f.replace('jpg', 'our_gp.npy')))

        our_op = get_capture_from_pc(our_op)
        assert our_op.shape != ()
        print(our_op.shape)
        # axs[i, 4].imshow(our_op)
        res[128 * 4:128 * 5, 416 * i: 416 * (i + 1)] = our_op

        sfm_op = get_capture_from_pc(sfm_op, False)
        assert sfm_op.shape != ()
        print(sfm_op.shape)
        # axs[i, 5].imshow(sfm_op)
        res[128 * 5:128 * 6, 416 * i: 416 * (i + 1)] = sfm_op

        gp = get_capture_from_pc(gp)
        assert gp.shape != ()
        print(gp.shape)
        # axs[i, 6].imshow(gp)
        res[128 * 6:, 416 * i: 416 * (i + 1)] = gp

        i += 1

    
    scipy.misc.toimage(res).save('temp.jpg')

    # fig = plt.figure(figsize=(n, 7))
    # plt.imshow(res)
    # plt.axis('off')
    # plt.show()

    # for ax in axs.flat:
    #     ax.set(xticks=[], yticks=[], aspect=1)

    # plt.tight_layout()#调整整体空白
    # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    # plt.show()




def presentation():
    pass


