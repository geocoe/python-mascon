import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from math import sin, cos
import pandas as pd
from cal_twsa_self import select_region_from_file
from netCDF4 import Dataset
import os
from tqdm import tqdm


# 构建系数矩阵
def coeff_matrix(Lon_dg, Lat_dg, Lon_x, Lat_x, reso, R):
    h = 480e3  # m
    r = R + h
    G = 6.67259e-11
    s = 510067866e6 / reso  # reso为地球表面格网个数

    Ax = np.zeros((len(Lon_dg), len(Lon_x)))
    Ay = np.zeros((len(Lon_dg), len(Lon_x)))
    Az = np.zeros((len(Lon_dg), len(Lon_x)))

    for i in range(len(Lon_dg)):
        for j in range(len(Lat_x)):
            x_x = R * np.cos(Lat_x[j]) * np.cos(Lon_x[j])
            y_x = R * np.cos(Lat_x[j]) * np.sin(Lon_x[j])
            z_x = R * np.sin(Lat_x[j])

            x_dg = r * np.cos(Lat_dg[i]) * np.cos(Lon_dg[i])
            y_dg = r * np.cos(Lat_dg[i]) * np.sin(Lon_dg[i])
            z_dg = r * np.sin(Lat_dg[i])

            lij = np.sqrt((x_dg - x_x) ** 2 + (y_dg - y_x) ** 2 + (z_dg - z_x) ** 2)

            Ax[i, j] = -G * 10.25 * s * (x_dg - x_x) / lij ** 3
            Ay[i, j] = -G * 10.25 * s * (y_dg - y_x) / lij ** 3
            Az[i, j] = -G * 10.25 * s * (z_dg - z_x) / lij ** 3

    A = np.vstack((Ax, Ay, Az))

    ncfile = Dataset('/home/master/shujw/CSRMASCON/outdata/world_co_matv3.nc', mode='w', format='NETCDF4')
    ncfile.createDimension('x', A.shape[0])
    ncfile.createDimension('y', A.shape[1])
    ewh_mask = ncfile.createVariable('co_mat', np.float32, ('x', 'y'))
    # 设置标题
    ncfile.title = 'ewh_mascon'
    ewh_mask[:, :] = A
    ncfile.close()
    print('系数矩阵已保存')

    return A


# 将等效水高、拐点索引数据保存为nc文件
def out_nc(mask, alpha, path=None, bln=None, basin_id=None):
    if basin_id != None:
        blnname = select_region_from_file(basin_id, bln)
    else:
        blnname = None
    if blnname != None:
        dirStr, _ = os.path.splitext(blnname)
        fname = dirStr.split("/")[-1]
        nc_name = f'{fname}_Lcurve_EWH.nc'
    else:
        nc_name = f'worldwide_Lcurve_EWH.nc'
    output_path = os.path.join(path, nc_name)
    if not output_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(output_path, mode='w', format='NETCDF4')

        ncfile.createDimension('len', mask.shape[0])
        ncfile.createDimension('time', mask.shape[1])
        ncfile.createDimension('alpha', len(alpha))

        ewh_mask = ncfile.createVariable('EWH', np.float32, ('len', 'time', 'alpha'))
        alp = ncfile.createVariable('alp', np.float32, ('alpha'))
        # 设置标题
        ncfile.title = 'ewh_mascon'
        ewh_mask.units = 'cm'

        ewh_mask[:, :, :] = mask
        alp[:] = alpha
        ncfile.close()

        print(f"等效水高已保存为{output_path}文件")


# 输出L曲线
def lcurve_png(path, bln, basin_id, text='PM'):
    if basin_id != 0:
        blnname = select_region_from_file(basin_id, bln)
    else:
        blnname = None
    if blnname != None:
        dirStr, _ = os.path.splitext(blnname)
        fname = dirStr.split("\\")[-1]
        # nc_name = f'{fname}_lcurve.png'
        nc_name = f'{fname}_lcurve_{text}'
    else:
        nc_name = f'worldwide_lcurve.png'
    output_path = os.path.join(path, nc_name)
    return output_path


# L曲线正则化求解
def tikhonov_lcuve(dg_vector, lon_dg, lat_dg, out_data_path, bln_path, t, rm=None):
    (num_mon, lenth) = dg_vector.shape
    a = 6378136.3
    lon_x = lon_dg.copy()  # , lon_x=None, lat_x=None
    lat_x = lat_dg.copy()
    # dataset = netCDF4.Dataset('/home/master/shujw/CSRMASCON/outdata/world_co_matv3.nc')
    # A = dataset['co_mat'][:]
    A = coeff_matrix(lon_dg, lat_dg, lon_x, lat_x, len(lon_dg), a)
    (m, n) = A.shape
    alpha = [1e-20, 1e-19, 5e-19, 7e-19, 9e-19, 1e-18, 3e-18, 1e-17, 1e-16]
    if rm is None:
        rm = np.eye(n)
    x_pm = np.zeros((n, num_mon, len(alpha)))
    v_dg = np.zeros((m, num_mon, len(alpha)))
    l_x = np.zeros((num_mon, len(alpha)))
    l_y = np.zeros((num_mon, len(alpha)))
    ATA = A.T @ A
    i = 0
    for alp in tqdm(alpha, desc='L曲线正则化EWH计算中'):
        # ATAbi = np.linalg.solve(ATA + alp * eye, A.T)
        ATAbi = np.linalg.solve(ATA + alp * rm, A.T)
        for j in range(num_mon):
            deta_g = np.zeros((lenth, 1))
            deta_g[:, 0] = dg_vector[j, :]
            # 计算解范数及残差范数
            x_pm[:, j, i] = np.matmul(ATAbi, deta_g)[:, 0]
            v_dg[:, j, i] = (np.matmul(A, x_pm[:, j, i]).reshape(-1, 1) - deta_g)[:, 0]
            l_x[j, i] = np.sum(v_dg[:, j, i] ** 2)
            l_y[j, i] = np.sum(x_pm[:, j, i] ** 2)
        i += 1

    l_x_sum = np.sum(l_x, axis=0) / num_mon
    l_y_sum = np.sum(l_y, axis=0) / num_mon

    out_nc(x_pm, alpha, out_data_path)  # 全球
    # 绘制L曲线
    fig = plt.figure(figsize=(20, 10), dpi=400)
    for i in range(num_mon):
        # for i in range(5):
        color = plt.cm.Blues((i + 1) / num_mon)
        plt.plot(l_x[i, :], l_y[i, :], 'b-', alpha=1, linewidth=1, color=color)
        # plt.scatter(l_x[i, :], l_y[i, :], marker='x')

    # 绘制平均L曲线
    plt.plot(l_x_sum, l_y_sum, 'r-', alpha=1, linewidth=1.5, label='Average L-curve')
    for i in range(len(alpha)):
        plt.scatter(l_x_sum[i], l_y_sum[i], marker='x', c='r', s=50, zorder=10)
        RP = '{:.2e}'.format(alpha[i])
        plt.annotate('RP={}'.format(RP), ha='left', va='bottom', xy=(l_x_sum[i], l_y_sum[i]),
                     c='k', fontsize=25)
    # 添加坐标轴说明
    plt.xlabel('||y-AX||$^{2}$$_{P}$')  # x_label
    plt.ylabel('||X||$^{2}$$_{M}$')  # y_label
    # 设置图例
    plt.legend(loc='best')
    PNG_NAME = lcurve_png(out_data_path, bln_path, t, )
    fig.savefig(PNG_NAME, dpi=400, bbox_inches='tight')

    return x_pm
