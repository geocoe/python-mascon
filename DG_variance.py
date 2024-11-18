import numpy as np
from calculating_DG import world_bln, read_region_bln
import read_gsm
import math
from associated_legendre import plm_holmes
from tqdm import tqdm
import bisect
from netCDF4 import Dataset
from calculating_DG import select_region_from_file
import os

# 求扣除均值后的球谐系数方差
def shc_variance(shc_var, time, start_time, end_time):
    count = len([x for x in time if end_time >= x >= start_time ])
    coe = np.zeros([len(time), len(time)])
    i = 0
    index0 = bisect.bisect_left(time, start_time)
    coe[:, index0:(count + index0)] = -(1/count)
    while i < len(time):
        if index0 <= i < index0 + count:
            coe[i, i] = (count - 1) / count
        else:
            coe[i, i] = 1
        i += 1
    cs_var = np.zeros_like(shc_var)
    for i in range(shc_var.shape[1]):
        for j in range(shc_var.shape[2]):
            for k in range(shc_var.shape[3]):
                cs_var_diag = np.diag(shc_var[:, i, j, k])
                cs_var_ori = coe @ cs_var_diag @ coe.T
                cs_var[:, i, j, k] = np.diag(cs_var_ori)
    return cs_var

# 将球谐系数方差传递到扰动重力
def var_propagation(cnm_var, snm_var, a, gm, lmax, lon, lat):
    r = a + 480e3
    kl = np.zeros((201, 1))
    temp = np.zeros((1, lmax + 1))
    with open(r"D:\ShareFile\MASCON\data\LoadLoveNumbers.txt") as fid:
        j = 0
        for content in fid:
            value = content.split()
            kl[j, 0] = float(value[0])
            j += 1
    for i in range(0, lmax + 1):
        temp[0, i] = (gm / (r ** 2)) * (i + 1) / (1 + kl[i, 0]) * ((a / r) ** i)
    cosmlambda = np.zeros((lmax + 1, lon.shape[0]))
    sinmlambda = np.zeros((lmax + 1, lon.shape[0]))
    for i in range(lmax + 1):
        for t in range(lon.shape[0]):
            cosmlambda[i, t] = math.cos(i * lon[t])
            sinmlambda[i, t] = math.sin(i * lon[t])
    lat = list(lat.flatten())
    Pnm0, _ = plm_holmes(lmax, np.sin(lat))
    row = len(lon)
    col = (lmax + 2) * (lmax + 1)
    coe = np.zeros((row, col))
    k = 0
    for i in range(lmax + 1):
        for j in range(i + 1):
            if k < col / 2:
                coe[:, int(i * (i + 1) / 2 + j)] = temp[0, i] * np.multiply(Pnm0[i, j, :], cosmlambda[j, :])
                coe[:, int(i * (i + 1) / 2 + j + col / 2)] = temp[0, i] * np.multiply(Pnm0[i, j, :], sinmlambda[j, :])
            k += 1
    shs_var = np.diag(np.concatenate((cnm_var[np.tril_indices(lmax + 1, k=0)], snm_var[np.tril_indices(lmax + 1, k=0)])))
    dg_var = coe @ shs_var @ coe.T
    tempp = np.linalg.cond(coe)
    return dg_var

def outnc(dg_var, bln, basin_id, save_path):
    if basin_id != 0:
        blnname = select_region_from_file(basin_id, bln)
    else:
        blnname = None
    if blnname != None:
        dirStr, _ = os.path.splitext(blnname)
        fname = dirStr.split("\\")[-1]
        nc_name = f'{fname}_dg_weight.nc'
    else:
        nc_name = f'worldwide_dg_weight.nc'
    output_path = os.path.join(save_path, nc_name)
    if not output_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(output_path, mode='w', format='NETCDF4')
        ncfile.createDimension('time', dg_var.shape[0])
        ncfile.createDimension('y', dg_var.shape[1])
        ncfile.createDimension('x', dg_var.shape[2])
        ewh_mask = ncfile.createVariable('dg_weight', np.float32, ('time', 'y', 'x'))
        # 设置标题
        ncfile.title = 'dg_weight'
        ewh_mask[:, :, :] = dg_var
        print(f"扰动重力权矩阵已保存为{output_path}文件")

'''求伪观测值（扰动重力）的方差协方差阵'''
def dg_variance(gsm_path, reso=1, bln=None, index=None, output_path=None):
    if index is None:
        lon, lat, _ = world_bln(reso)
    else:
        lon, lat, region, _ = read_region_bln(bln, index, reso)
    # 读取球谐系数标准差
    _, shc_std, time, _, lmax, a, gm = read_gsm.read_all_gsm(gsm_path)
    shc_var = shc_std ** 2
    # 对球谐系数方差阵做扣除均值处理的传递
    shc_ano_var = shc_variance(shc_var, time, 2004, 2010)
    dg_var = []
    # 基于扰动重力计算公式计算伪观测值的权阵
    for clm_var in tqdm(shc_ano_var, desc='扰动重力方差阵计算中'):
        # 分离C、S系数
        cnm_var = clm_var[0, :, :]
        snm_var = clm_var[1, :, :]
        # 计算重力扰动
        dg_var0 = var_propagation(cnm_var, snm_var, a, gm, lmax, lon, lat)
        dg_var.append(dg_var0)
    dg_var = np.array(dg_var)
    # 保存扰动重力权矩阵
    lenth = int(dg_var.shape[0])
    dg_weight = np.linalg.inv(dg_var[:lenth, :, :])
    outnc(dg_weight, bln, index, output_path)
    return dg_weight