import numpy as np
import sys
from scipy.interpolate import interp1d


def lovenums(l):
    '''
    Estimate Load Love Numbers
    l: Degree of Load Love Numbers
    '''
    k_l_list = np.array(
        [[0, 0.000], [1, 0.021], [2, -0.303], [3, -0.194], [4, -0.132], [5, -0.104], [6, -0.089], [7, -0.081],
         [8, -0.076], [9, -0.072], [10, -0.069], [12, -0.064], [15, -0.058], [20, -0.051], [30, -0.040],
         [40, -0.033], [50, -0.027], [70, -0.020], [100, -0.014], [150, -0.010], [200, -0.007]])

    f = interp1d(k_l_list[:, 0], k_l_list[:, 1])
    lln = f(l)
    return lln


def shc2mc(shc, equi_material='Water', lmax=60):
    '''
    Converts geoid coefficients to mass coefficients
    '''
    if equi_material == 'Water':
        rho = 1000
    elif equi_material == 'Ice':
        rho = 917
    elif equi_material == 'Sand':
        rho = 1442
    else:
        sys.exit('输入的密度未知，程序退出')

    # Calculate the average density of the Earth
    G = 6.67430e-11
    GM = 3.986004415e14
    a = 6378136.3
    rho_ave = 5517

    mc = np.zeros_like(shc)
    # mc_std = np.zeros_like(shc_std)

    for l in range(lmax + 1):
        l = int(l)
        lln = lovenums(l)
        factor = a * rho_ave / (3 * rho) * (2 * l + 1) / (1 + lln)
        mc[:, :, l, :] = factor * shc[:, :, l, :]
        # if shc_std == None:
        #     continue
        # else:
        #     mc_std[:, :, l, :] = factor * shc_std[:, :, l, :]

    return mc


def mc2shc(mclm, mslm, shcname):
    # 初始化地球物理参数
    a = 6378136.3
    rho_ave = 5517
    if 'CSR_mas' in shcname:
        rho = 1025
    else:
        rho = 1000

    LMAX, MMAX = mclm.shape
    l = np.arange(0, LMAX, 1)
    lln = lovenums(l)
    factor = a * rho_ave * (2 * l + 1) / (3 * rho * (1 + lln))
    factor = factor.reshape((factor.shape[0],1))
    shclm = mclm / factor
    shslm = mslm / factor
    # 将cs系数从两个矩阵合并为一个矩阵C\S
    for m in range(1, LMAX, 1):
        shclm[m - 1, m:LMAX] = shslm[m:LMAX, m]
    return shclm
