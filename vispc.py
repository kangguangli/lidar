import numpy as nu
from open3d import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import time



def visualize_statistics(file : str):

    arr = np.fromfile(file, dtype = 'float32')
    arr = arr.reshape((-1, 4))
    arr = arr[:, :3]

    # x = pd.Series(arr[:, 0])
    # y = pd.Series(arr[:, 1])
    # z = pd.Series(arr[:, 2])

    # x.plot.hist()


def get_max_min_from_file(file : str):

    arr = np.fromfile(file, dtype = 'float32')
    arr = arr.reshape((-1, 4))
    arr = arr[:, :3]

    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]

    return x.max(), x.min(), y.max(), y.min(), z.max(), z.min()



def distribution_single_file(file : str):

    bins = 20

    arr = np.fromfile(file, dtype = 'float32')
    arr = arr.reshape((-1, 4))
    arr = arr[:, :3]

    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]

    x_max, x_min, y_max, y_min, z_max, z_min = x.max(), x.min(), y.max(), y.min(), z.max(), z.min()

    x_distribution, x_intervals = np.histogram(x, bins = bins, range = [x_min, x_max])
    y_distribution, y_intervals = np.histogram(y, bins = bins, range = [y_min, y_max])
    z_distribution, z_intervals = np.histogram(z, bins = bins, range = [z_min, z_max])

    plt.figure(3)

    plt.subplot(131)
    plt.bar(x_intervals[:-1], x_distribution)
    plt.xlim(x_min, x_max)

    plt.subplot(132)
    plt.bar(y_intervals[:-1], y_distribution)
    plt.xlim(y_min, y_max)

    plt.subplot(133)
    plt.bar(z_intervals[:-1], z_distribution)
    plt.xlim(z_min, z_max)

    plt.show()
    # fig = plt.gcf()
    # fig.savefig(file.replace('bin', 'png'), format='png', transparent=True)



def distribution_statistics(dir : str = 'data/velo', postfix = 'bin'):

    files = os.listdir(dir)
    files = [x for x in files if x.find(postfix) != -1]

    x_max = -1e10
    x_min = 1e10
    y_max = -1e10
    y_min = 1e10
    z_max = -1e10
    z_min = 1e10

    for f in files:
        _x_max, _x_min, _y_max, _y_min, _z_max, _z_min = get_max_min_from_file(os.path.join(dir, f))
        x_max = max(x_max, _x_max)
        x_min = min(x_min, _x_min)
        y_max = max(y_max, _y_max)
        y_min = min(y_min, _y_min)
        z_max = max(z_max, _z_max)
        z_min = min(z_min, _z_min)

    bins = 20

    z_distribution = y_distribution = x_distribution = np.zeros(bins)
    x_intervals = np.linspace(x_min, x_max, bins + 1)
    y_intervals = np.linspace(y_min, y_max, bins + 1)
    z_intervals = np.linspace(z_min, z_max, bins + 1)

    for f in files:
        arr = np.fromfile(os.path.join(dir, f), dtype = 'float32')
        arr = arr.reshape((-1, 4))
        arr = arr[:, :3]

        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]

        x_distribution += np.histogram(x, bins = bins, range = [x_min, x_max])[0]
        y_distribution += np.histogram(y, bins = bins, range = [y_min, y_max])[0]
        z_distribution += np.histogram(z, bins = bins, range = [z_min, z_max])[0]
    # plt.bar(x_intervals[:-1], x_distribution, width = 1)
    # plt.xlim(x_min, x_max)

    plt.figure(3)

    plt.subplot(131)
    plt.bar(x_intervals[:-1], x_distribution, width = 1)
    # plt.xlim(x_min, x_max)

    plt.subplot(132)
    plt.bar(y_intervals[:-1], y_distribution, width = 1)
    # plt.xlim(y_min, y_max)

    plt.subplot(133)
    plt.bar(z_intervals[:-1], z_distribution, width = 1)
    # plt.xlim(z_min, z_max)

    #plt.show()
    fig = plt.gcf()
    fig.savefig(str(time.asctime(time.localtime(time.time()))), format='png', transparent=True)


def readVariable(file : str) :

    with open(file, 'r') as f:

        lines = f.readlines()
        lines = [x.strip() for x in lines if x != '\n']
        assert len(lines) == 7

        lines = list(map(lambda x : x.split(' '), lines))

        p0 = np.matrix(np.array(lines[0][1:], dtype = np.float32).reshape((3, 4)))
        p1 = np.matrix(np.array(lines[1][1:], dtype = np.float32).reshape((3, 4)))
        p2 = np.matrix(np.array(lines[2][1:], dtype = np.float32).reshape((3, 4)))
        p3 = np.matrix(np.array(lines[3][1:], dtype = np.float32).reshape((3, 4)))

        r0_rect = np.array(lines[4][1:], dtype = np.float32).reshape((3, 3))
        r0_rect = np.pad(r0_rect, ((0, 1), (0, 1)), mode = 'constant', constant_values = 0)
        r0_rect[-1][-1] = 1
        r0_rect = np.matrix(r0_rect)
        assert r0_rect.shape == (4, 4)

        tr_velo_to_cam = np.array(lines[5][1:], dtype = np.float32).reshape((3, 4))
        tr_velo_to_cam = np.pad(tr_velo_to_cam, ((0, 1), (0, 0)), mode = 'constant', constant_values = 0)
        tr_velo_to_cam[-1][-1] = 1
        tr_velo_to_cam = np.matrix(tr_velo_to_cam)
        assert tr_velo_to_cam.shape == (4, 4)

        tr_imu_to_velo = np.array(lines[5][1:], dtype = np.float32).reshape((3, 4))
        tr_imu_to_velo = np.pad(tr_imu_to_velo, ((0, 1), (0, 0)), mode = 'constant', constant_values = 0)
        tr_imu_to_velo[-1][-1] = 1
        tr_imu_to_velo = np.matrix(tr_imu_to_velo)
        assert tr_imu_to_velo.shape == (4, 4)

        return p0, p1, p2, p3, r0_rect, tr_velo_to_cam, tr_imu_to_velo



def project_to_image(file : str, pc_dir = 'data/velo', calib_dir = 'data/data_object_calib/training/calib'):

    if file.find('bin') == -1:
        return

    _, _, p2, p3, r0_rect, tr_velo_to_cam, _ = readVariable(os.path.join(calib_dir, file.replace('bin', 'txt')))

    # f = lambda p : p2 * r0_rect * tr_velo_to_cam * np.transpose(np.matrix(p))
    # f = np.vectorize(f)

    # arr = np.fromfile(os.path.join(pc_dir, file), dtype = 'float32')
    # arr = arr.reshape((-1, 4))
    # arr = arr[:, :3]

    arr = getArray(os.path.join(pc_dir, file), four = True)

    #filter no use
    # selector = np.vectorize(lambda x: True if x  else False)

    new_pc = np.transpose(p2 * r0_rect * tr_velo_to_cam * np.transpose(arr))

    # new_pc = f(arr)

    return new_pc