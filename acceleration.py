import numpy as np
from associated_legendre import plm_holmes
from read_gsm import read_all_gsm
from gsm_deaverage import remove_baseline
from replace_SHs import replace_shc
from gc2mc import lovenums
from grace_time import date2doy
from harmonic_summation import cs_separate

def acceleration(gravity_models, tn13, tn14, region_lons, region_lats):
    # 初始化参数
    r = 6378136.3
    gm = 3.986004415e14
    h = 480e3
    lon = np.deg2rad(region_lons)
    lat = np.deg2rad(region_lats)
    three_dimension_accelerations,model_names = [], []
    for model in gravity_models:
        cs = model['data']
        date = model['date']
        maxDegree = model['degree']
        name = model['name']
        model_names.append(name)
        time = date2doy(date)
        num_mon = len(time)
        # 转化球谐系数的存储形式
        cs = cs_separate(cs)
        # 进行C10，C11，S11，C20，C30替换
        cs_replace = replace_shc(cs, time, tn13, tn14)

        # 扣除2004年至2009年的平均值（2004-2009）
        cs_anomy = remove_baseline(cs_replace, time, 2004, 2010)
        # 勒夫负荷系数
        l = np.arange(0, maxDegree + 1, 1)
        kl = lovenums(l)
        factor = np.zeros(maxDegree + 1)
        for n in range(maxDegree + 1):
            factor[n] = (gm / (2 * (r + h) ** 2)) * np.sqrt((2 * n + 1) / (2 * n + 3)) / (1 + kl[n]) * (
                    r / (h + r)) ** n
        delta1m = np.zeros(maxDegree + 1)
        delta1m[1] = 1
        delta0m = np.zeros(maxDegree + 1)
        delta0m[0] = 1
        f1 = np.zeros((maxDegree + 1, maxDegree + 1))
        f2 = np.zeros((maxDegree + 1, maxDegree + 1))
        f3 = np.zeros((maxDegree + 1, maxDegree + 1))
        for n in range(maxDegree + 1):
            for m in range(n + 1):
                f1[n, m] = np.sqrt((n - m + 1) * (n - m + 2) * (1 + delta1m[m]))
                f2[n, m] = np.sqrt((n - m + 1) * (n + m + 1))
                f3[n, m] = np.sqrt((n + m + 1) * (n + m + 2) * (1 + delta0m[m]))
        # 构造勒让德函数
        lat = np.mat(list(lat.flatten()))
        lat = list(lat.flatten())
        Pnm, _ = plm_holmes(maxDegree + 1, np.sin(lat))
        P1 = np.zeros((maxDegree + 1, maxDegree + 1, len(lon)))
        P2 = np.zeros((maxDegree + 1, maxDegree + 1, len(lon)))
        P3 = np.zeros((maxDegree + 1, maxDegree + 1, len(lon)))
        for n in range(maxDegree + 1):
            for m in range(n + 1):
                P2[n, m, :] = Pnm[n + 1, m, :]
                P3[n, m, :] = Pnm[n + 1, m + 1, :]
            for m in range(1, n + 1):
                P1[n, m, :] = Pnm[n + 1, m - 1, :]
        lon = np.mat(list(lon.flatten()))
        cosm1 = np.cos(np.dot((np.arange(maxDegree + 1)[:, None] - 1), lon))
        cosm2 = np.cos(np.dot(np.arange(maxDegree + 1)[:, None], lon))
        cosm3 = np.cos(np.dot((np.arange(maxDegree + 1)[:, None] + 1), lon))
        sinm1 = np.sin(np.dot((np.arange(maxDegree + 1)[:, None] - 1), lon))
        sinm2 = np.sin(np.dot(np.arange(maxDegree + 1)[:, None], lon))
        sinm3 = np.sin(np.dot((np.arange(maxDegree + 1)[:, None] + 1), lon))
        C1 = f1[:, :, None] * P1
        C2 = f2[:, :, None] * P2
        C3 = f3[:, :, None] * P3
        S1 = f1[:, :, None] * P1
        S2 = f2[:, :, None] * P2
        S3 = f3[:, :, None] * P3
        lon = np.array(lon).flatten()
        lat = np.array(lat).flatten()
        ax = np.zeros((num_mon, len(lat)))
        ay = np.zeros((num_mon, len(lat)))
        az = np.zeros((num_mon, len(lat)))
        cnm = cs_anomy[:, 0, :, :]
        snm = cs_anomy[:, 1, :, :]
        for k in range(num_mon):
            for i in range(len(lat)):
                ax[k, i] = factor * (
                        cnm[k, :, :] * C1[:, :, i] * cosm1[:, i] - cnm[k, :, :] * C3[:, :, i] * cosm3[:, i] + (
                        snm[k, :, :] * S1[:, :, i] * sinm1[:, i] - snm[k, :, :] * S3[:, :, i] * sinm3[:, i]))
                ay[k, i] = factor * (
                        -cnm[k, :, :] * S1[:, :, i] * sinm1[:, i] - cnm[k, :, :] * S3[:, :, i] * sinm3[:, i] + (
                        snm[k, :, :] * C1[:, :, i] * cosm1[:, i] + snm[k, :, :] * C3[:, :, i] * cosm3[:, i]))
                az[k, i] = factor * (
                        -2 * cnm[k, :, :] * C2[:, :, i] * cosm2[:, i] - 2 * snm[k, :, :] * S2[:, :, i] * sinm2[:,
                                                                                                         i])

        a_xyz = np.concatenate((ax, ay, az), axis=1)
        three_dimension_accelerations.append(a_xyz)
    three_dimension_accelerations = np.array(three_dimension_accelerations)


    return three_dimension_accelerations, time, model_names
