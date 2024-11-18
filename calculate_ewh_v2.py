import sys
import numpy as np
from tqdm import tqdm
from read_gsm import read_all_gsm
from gsm_deaverage import remove_baseline
from replace_SHs import replace_shc
import os
from netCDF4 import Dataset
import netCDF4 as nc
from associated_legendre import plm_holmes
import matplotlib.pyplot as plt
from l_curve_regular import lcurve_png
import pdb


# 构建系数矩阵
def coeff_matrix(lon_x, lat_x, reso, gm, a, lmax):
    G = 6.67259e-11
    earth_mass = gm / G
    s = 4 * np.pi / reso
    maxdegree = lmax
    # 读取负荷勒夫数
    kl = np.zeros(201)
    longitude = np.mat(list(lon_x.flatten()))
    latitude = np.mat(list(lat_x.flatten()))
    with open(r"/home/master/shujw/CSRMASCON/data/LoadLoveNumbers.txt") as fid:
        j = 0
        for content in fid:
            value = content.split()
            kl[j] = float(value[0])
            j = j + 1

    # 计算三角函数值
    m = np.arange(0, maxdegree + 1)[:, np.newaxis]
    ccos = np.cos(np.dot(m, longitude))
    ssin = np.sin(np.dot(m, longitude))

    # 计算勒让德函数
    lat = list(latitude.flatten())
    Pnm0, _ = plm_holmes(maxdegree, np.sin(lat))
    costant1 = np.zeros(maxdegree + 1)
    for i in range(maxdegree + 1):
        costant1[i] = 10.25 * (1 + kl[i]) * a ** 2 / (earth_mass * (2 * i + 1)) * s
    pdb.set_trace()
    col = (maxdegree + 2) * (maxdegree + 1)
    coe = np.zeros((col, latitude.shape[1]))
    k = 0
    for i in range(maxdegree + 1):
        for j in range(i + 1):
            if k < col / 2:
                coe[int(i * (i + 1) / 2 + j), :] = costant1[i] * np.multiply(Pnm0[i, j, :], ccos[j, :])
                coe[int(i * (i + 1) / 2 + j + col / 2), :] = costant1[i] * np.multiply(Pnm0[i, j, :], ssin[j, :])
            k += 1

    # ncfile = Dataset(r'/home/master/QK/pointmass/results/world_co_matv2.nc', mode='w', format='NETCDF4')
    # ncfile.createDimension('x', coe.shape[0])
    # ncfile.createDimension('y', coe.shape[1])
    # ewh_mask = ncfile.createVariable('co_mat', np.float64, ('x', 'y'))
    # # 设置标题
    # ncfile.title = 'ewh_mascon'
    # ewh_mask[:, :] = coe
    # ncfile.close()
    # print('系数矩阵已保存')

    return coe


# 保存输出结果为nc文件
def out_nc(dg, alpha, path=None):
    nc_name = f'worldwideEWH_yueshu.nc'
    output_path = os.path.join(path, nc_name)
    if not output_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(output_path, mode='w', format='NETCDF4')

        ncfile.createDimension('long', dg.shape[0])
        ncfile.createDimension('time', dg.shape[1])
        ncfile.createDimension('len', len(alpha))

        DG = ncfile.createVariable('EWH', np.float32, ('long', 'time', 'len'))
        alp = ncfile.createVariable('alp', np.float32, ('len',))
        # 设置标题
        ncfile.title = 'disturb_gravity'

        DG[:, :, :] = dg
        alp[:] = alpha
        ncfile.close()

        print(f"等效水高结果已保存为{output_path}文件")


# 计算区域重力扰动/默认全球
def ewh_v2(gsm_path, tn13, tn14, reso=1, bln=None, output_path=None, rm=None):
    # 计算选定区域扰动重力
    # 设定TWSA分辨率，目前可选0.25，0.5，和1
    dataset = nc.Dataset(r'/home/master/shujw/CSRMASCON/data/Lon_Lat_vec.nc')
    lon = np.deg2rad(dataset.variables["lon"][:])
    lat = np.deg2rad(dataset.variables["lat"][:])
    # 读取GSM文件
    cs, _, time, _, lmax, r, gm = read_all_gsm(gsm_path)

    # 进行C10，C11，S11，C20，C30替换
    cs_replace = replace_shc(cs, time, tn13, tn14)

    # 扣除2004年至2009年的平均值（2004-2009）
    cs_anomy = remove_baseline(cs_replace, time, 2004, 2010)

    # 设置正则化参数取值范围
    # alpha = [6e3, 8e3, 9e3, 1e4, 1.5e4, 2e4, 2.5e4, 3e4, 3.5e4, 4e4, 4.5e4, 5e4]
    # alpha = [1e10, 1e11, 1e12, 5e12, 1e13, 5e13, 1e14, 1e15]
    # alpha = [1e-26, 2e-26, 3e-26, 4e-26, 5e-25, 7e-26, 8.5e-26, 1e-25] # 无约束
    # alpha = [5e-25, 8e-25, 1e-24, 2e-24, 3e-24, 4e-24, 5e-24, 6e-24, 7e-24, 8e-24, 9e-24, 1e-23, 3e-23, 7e-23]
    alpha = [1e-25]
    # 构建系数矩阵
    A = coeff_matrix(lon, lat, lon.shape[0], gm, r, lmax)

    (m, n) = A.shape
    num_mon = cs_anomy.shape[0]
    if rm is None:
        rm = np.eye(n)
    x_pm = np.zeros((n, 1, len(alpha)))
    ATA = np.dot(A.T, A)
    # pdb.set_trace()
    i = 0
    for alp in tqdm(alpha, desc='L曲线正则化EWH计算中'):
        # ATAbi = np.linalg.solve(ATA + alp * rm, A.T)
        for j in range(1):
            # 构建观测值
            cnm = cs_anomy[j, 0, :, :]
            snm = cs_anomy[j, 1, :, :]
            y = np.concatenate((cnm[np.tril_indices(lmax + 1, k=0)], snm[np.tril_indices(lmax + 1, k=0)]))
            y = y[:, np.newaxis]
            b = np.dot(A.T, y)
            # zero_positions = np.where(y == 0)
            # y = np.delete(y, zero_positions[0], axis=0)
            # 计算解范数及残差范数
            # x_pm[:, j, i] = np.dot(ATAbi, y)[:, 0]
            N = ATA + alp * rm
            pdb.set_trace()
            x = np.linalg.solve(N, b)
            # x = np.dot(ATAbi, y)[:, 0]
            x_pm[:, j, i] = np.squeeze(x)
        i += 1

    out_nc(x_pm, alpha, output_path)  # 全球

    return x_pm, time
