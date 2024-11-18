import sys

import numpy as np


def filter_gaussian(shc, r=0, a=6371):
    '''
    Gaussian filter used to attenuate noise
    input r : Gaussian filer radius in km
    filt_SHC = filt_gaussian(SHC,r = 300)
    filt_SHC,filt_SHC_std = filt_gaussian(SHC,SHC_std,r = 300)
    '''
    if r < 0:
        sys.exit('未输入有效滤波半径')
    elif r == 0:
        print('未进行高斯滤波')
        filter_shc = shc
    else:
        # print(f"进行 {r} km 半径的高斯滤波")
        n = shc.shape[-1] - 1
        b = np.log(2) / (1 - np.cos(r / a))
        W = np.ones(n + 1)
        W[1] = (1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b)) - 1 / b

        # 递归公式推导高斯滤波权重
        for i in range(1, n):
            W[i + 1] = -(2 * i + 1) / b * W[i] + W[i - 1]

        filter_shc = shc * W[:, None]

    return filter_shc
