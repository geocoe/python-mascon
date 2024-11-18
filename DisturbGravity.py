"""
计算重力扰动
input:
output:
Input:弧度下的经纬度列向量（px1,qx1)、球谐系数、地心引力常数、地球参考半径
Output:矩阵(qxp)存储的扰动重力格网
"""
import numpy as np
import math
from associated_legendre import plm_holmes
from gc2mc import lovenums
from grace_time import date2doy
from harmonic_summation import cs_separate
from gsm_deaverage import remove_baseline


def disturbgravity(gravity_models, region_lons, region_lats):
    a = 6378136.3
    gm = 3.986004415e14
    h = 480e3
    r = h + a
    longitude = np.deg2rad(region_lons)
    latitude = np.squeeze(np.deg2rad(region_lats))

    num_observation = len(longitude)
    dgs, model_names = [], []

    for model in gravity_models:
        cs = model['data']
        date = model['date']
        maxdegree = model['degree']
        name = model['name']
        model_names.append(name)
        time = date2doy(date)
        num_mon = len(time)
        # 转化球谐系数的存储形式
        cs = cs_separate(cs)
        # 扣除2004年至2009年的平均值（2004-2009）
        cs_anomy = remove_baseline(cs, time, 2004, 2010)
        cnm = cs_anomy[:, 0, :, :]
        snm = cs_anomy[:, 1, :, :]

        # 勒夫负荷系数
        l = np.arange(0, maxdegree + 1, 1)
        kl = lovenums(l)

        # 计算尺度因子
        factor = np.zeros((1, maxdegree + 1))
        for i in range(0, maxdegree + 1):
            factor[0, i] = (gm / (r ** 2)) * (i + 1) / (1 + kl[i]) * ((a / r) ** i)
        # 计算三角函数值
        m = np.arange(0, maxdegree + 1)[:, np.newaxis]
        lamda=longitude.T
        cosmlambda = np.cos(np.dot(m, lamda))
        sinmlambda = np.sin(np.dot(m, lamda))

        dg = np.zeros((num_mon, num_observation))
        # 计算勒让德函数
        Pnm0, _ = plm_holmes(maxdegree, np.sin(latitude))

        for i in range(num_mon):
            for j in range(num_observation):
                dg[i, j] = factor @ (
                        Pnm0[:, :, j] * cnm[i, :, :] @ cosmlambda[:, j] + Pnm0[:, :, j] * snm[i, :, :] @ sinmlambda[
                                                                                                         :, j])
        dgs.append(dg)
    dgs = np.array(dgs)

    return dgs, time, model_names
