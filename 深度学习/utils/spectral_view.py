# create by andy at 2022/4/18
# reference: 

import numpy as np
from spectral import save_rgb


def show(name, array, bands):
    save_rgb(name, array, bands)


if __name__ == '__main__':
    name = '/media/andy/z/python/graduation_design/深度学习/Data/obt/train/Data/0005.npy'
    HSI = np.load(name, allow_pickle=True).item().get("image")
    show('rgb.jpg', HSI, [29, 19, 31])
