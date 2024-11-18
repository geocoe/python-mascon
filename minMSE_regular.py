import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos
import pandas as pd
from calculating_DG import select_region_from_file
from netCDF4 import Dataset
import os
from tqdm import tqdm
from min_mse import *
import loaddata


# 构建系数矩阵
def coeff_matrix(lon_dg, lat_dg, lon_x, lat_x, reso, a):
    h = 480e+3
    r = a + h
    G = 6.67259e-11
    reso = 64800
    s = 510067866e+6 / reso
    # lon = np.deg2rad(lon_dg)
    # lat = np.deg2rad(lat_dg)
    A = np.zeros((lon_dg.shape[0], lon_x.shape[0]), dtype=np.float32)
    for i in range(lon_dg.shape[0]):
        for j in range(lat_x.shape[0]):
            cospsi = sin(lat_x[i]) * sin(lat_dg[j]) + cos(lat_x[i]) * cos(lat_dg[j]) * cos(lon_x[i] - lon_dg[j])
            A[i, j] = G * 10 * s * (r - a * cospsi) / (a ** 2 + r ** 2 - 2 * a * r * cospsi) ** 1.5
    return A

# 将等效水高、拐点索引数据保存为nc文件
def out_nc(mask, minMSEalpha, path=None, bln=None, basin_id=None):
    if basin_id != 0:
        blnname = select_region_from_file(basin_id, bln)
    else:
        blnname = None
    if blnname != None:
        dirStr, _ = os.path.splitext(blnname)
        fname = dirStr.split("/")[-1]
        nc_name = f'{fname}_minMSE_EWH.nc'
    else:
        nc_name = f'worldwide_minMSE_EWH.nc'
    output_path = os.path.join(path, nc_name)
    if not output_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(output_path, mode='w', format='NETCDF4')

        ncfile.createDimension('x', mask.shape[0])
        ncfile.createDimension('y', mask.shape[1])
        ncfile.createDimension('time', mask.shape[2])

        ewh_mask = ncfile.createVariable('EWH', np.float32, ('x', 'y', 'time'))
        minMSE_alpha = ncfile.createVariable('RP', np.float32, ('time'))
        # 设置标题
        ncfile.title = 'ewh_mascon'

        ewh_mask.units = 'cm'

        ewh_mask[:, :, :] = mask
        minMSE_alpha[:] = minMSEalpha
        ncfile.close()

        print(f"等效水高已保存为{output_path}文件")


def tikhonov_minMSE(dg_vector1, lon_dg, lat_dg, mask, out_data_path, bln_path, t):
    num_mon = dg_vector1.shape[2]
    a = 6378136.3
    lon_x = lon_dg.copy()  # , lon_x=None, lat_x=None
    lat_x = lat_dg.copy()
    A = coeff_matrix(lon_dg, lat_dg, lon_x, lat_x, len(lon_dg), a)
    (m, n) = A.shape
    # alpha = [1e-28, 1e-26, 1e-25, 1e-20, 7e-13, 1e-9, 1e-5, 1e-2]
    eye = np.eye(n)
    ATA = A.T @ A
    minMSE_alpha = np.zeros(num_mon)
    x_mse = np.zeros((A.shape[1], num_mon))
    dg_vector = np.transpose(dg_vector1, axes=(2, 0, 1))
    # 读取l曲线正则化结果
    ewh_l, index = loaddata.loadTWH(0, out_data_path, bln_path, t)

    i = 0
    for dg in tqdm(dg_vector, desc='minMSE正则化EWH计算中'):
        dg = np.roll(dg, 180, axis=1)
        detag = dg.reshape(-1, 1)
        temp = pd.DataFrame(detag)
        temp = temp.dropna()
        deta_g = np.asarray(temp)

        # 将l曲线正则化结果转化为列向量
        temp = np.roll(ewh_l[:, :, i], 180, axis=1)
        temp = temp.reshape(-1, 1)
        temp = pd.DataFrame(temp)
        temp = temp.dropna()
        x_lcur = np.asarray(temp)
        # 求l曲线正则化解的无偏验后估计中误差
        sigma2 = sigma2_tik(deta_g, A, index[i], x_lcur)

        # 求最小MSE正则化参数
        minMSE_alpha[i] = minMSE_NIT(A, sigma2, x_lcur, 1e-20)
        x_mse[:, i] = (np.linalg.solve((ATA + minMSE_alpha[i] * eye), A.T @ deta_g))[:, 0]
        i += 1

    # 将等效水高数据输入掩膜矩阵
    ewh_mask = np.repeat(mask[:, :, np.newaxis], num_mon, axis=2)
    m, n = np.where(~np.isnan(mask))
    for i in range(num_mon):
        for k in range(x_mse.shape[0]):
            ewh_mask[m[k], n[k], i] = x_mse[k, i]
    plt.show()
    # 将计算结果保存为nc文件
    out_nc(ewh_mask, minMSE_alpha, out_data_path, bln_path, t)
    return ewh_mask
