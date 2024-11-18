from associated_legendre import plm_holmes
from gc2mc import lovenums
import numpy as np


def disturb_gravity_covariance(cnm_var, snm_var, lmax, lon, lat):
    nlon= len(lon)
    # 度转弧度
    phi = np.deg2rad(lon)[np.newaxis,:]
    th = np.deg2rad(lat)
    a = 6378136.3
    r = a + 480e3
    gm = 3.986004415e14
    temp = np.zeros((1, lmax + 1))

    # 勒夫负荷系数
    l = np.arange(0, lmax + 1, 1)
    kl = lovenums(l)

    # 系数转换尺度因子
    for i in range(0, lmax + 1):
        temp[0, i] = (gm / (r ** 2)) * (i + 1) / (1 + kl[i]) * ((a / r) ** i)

    # 计算三角函数值
    m = np.arange(0, lmax + 1)[:, np.newaxis]
    ccos = np.cos(np.multiply(m, phi))
    ssin = np.sin(np.multiply(m, phi))

    # 计算勒让德函数
    Pnm0, _ = plm_holmes(lmax, np.sin(th))

    # 构建系数矩阵
    col = (lmax + 2) * (lmax + 1)
    coe = np.zeros((nlon, col))
    k = 0
    for i in range(lmax + 1):
        for j in range(i + 1):
            if k < col // 2:
                coe[:, i * (i + 1) // 2 + j] = temp[0, i] * np.multiply(Pnm0[i, j, :], ccos[j, :])
                coe[:, i * (i + 1) // 2 + j + col // 2] = temp[0, i] * np.multiply(Pnm0[i, j, :], ssin[j, :])
            k += 1

    # 误差传播到重力扰动位的协方差阵
    shs_var = np.diag(
        np.concatenate((cnm_var[np.tril_indices(lmax + 1, k=0)], snm_var[np.tril_indices(lmax + 1, k=0)])))
    dg_var = coe @ shs_var @ coe.T
    tempp = np.linalg.cond(dg_var)
    print(f'重力扰动位的协方差阵条件数为:{tempp}')
    return dg_var

def disturb_gravity_covariance1(cnm_var, snm_var, lmax, lon, lat):
    nlon= len(lon)
    # 度转弧度
    phi = np.deg2rad(lon)[np.newaxis,:]
    th = np.deg2rad(lat)
    a = 6378136.3
    r = a + 480e3
    gm = 3.986004415e14
    temp = np.zeros((1, lmax + 1))

    # 勒夫负荷系数
    l = np.arange(0, lmax + 1, 1)
    kl = lovenums(l)

    # 系数转换尺度因子
    for i in range(0, lmax + 1):
        temp[0, i] = (gm / (r ** 2)) * (i + 1) / (1 + kl[i]) * ((a / r) ** i)

    # 计算三角函数值
    m = np.arange(0, lmax + 1)[:, np.newaxis]
    ccos = np.cos(np.multiply(m, phi))
    ssin = np.sin(np.multiply(m, phi))

    # 计算勒让德函数
    Pnm0, _ = plm_holmes(lmax, np.sin(th))

    # 构建系数矩阵
    col = (lmax + 2) * (lmax + 1)
    coe = np.zeros((nlon, col))
    k = 0
    for i in range(lmax + 1):
        for j in range(i + 1):
            if k < col // 2:
                coe[:, i * (i + 1) // 2 + j] = temp[0, i] * np.multiply(Pnm0[i, j, :], ccos[j, :])
                coe[:, i * (i + 1) // 2 + j + col // 2] = temp[0, i] * np.multiply(Pnm0[i, j, :], ssin[j, :])
            k += 1

    # 误差传播到重力扰动位的协方差阵
    shs_var = np.concatenate((cnm_var[np.tril_indices(lmax + 1, k=0)], snm_var[np.tril_indices(lmax + 1, k=0)]))
    non_zero_index = shs_var.nonzero()[0]
    shs_var[non_zero_index]=1.0/shs_var[non_zero_index]
    H=coe@np.diag(np.sqrt(shs_var))
    # 去除矩阵为0的列
    non_zero_columns = (H.sum(axis=0)!=0)
    H = H[:, non_zero_columns]


    return H