import numpy as np
from gc2mc import lovenums
from associated_legendre import plm_holmes
from math import sin, cos
def coeff_matrix(Lon_dg, Lat_dg, Lon_x, Lat_x, reso):
    # 初始化参数
    h = 480e3  # m
    R = 6378136.3
    r = R + h
    G = 6.67259e-11
    s = 510067866e6 / reso  # reso为地球表面格网个数

    Ax = np.zeros((len(Lon_dg), len(Lon_x)))
    Ay = np.zeros((len(Lon_dg), len(Lon_x)))
    Az = np.zeros((len(Lon_dg), len(Lon_x)))

    for i in range(len(Lon_dg)):
        for j in range(len(Lat_x)):
            # 质量块的直角三维坐标
            x_x = R * np.cos(Lat_x[j]) * np.cos(Lon_x[j])
            y_x = R * np.cos(Lat_x[j]) * np.sin(Lon_x[j])
            z_x = R * np.sin(Lat_x[j])

            # 空间点的直角三维坐标
            x_dg = r * np.cos(Lat_dg[i]) * np.cos(Lon_dg[i])
            y_dg = r * np.cos(Lat_dg[i]) * np.sin(Lon_dg[i])
            z_dg = r * np.sin(Lat_dg[i])

            # 计算空间点和质量块之间的距离
            lij = np.sqrt((x_dg - x_x) ** 2 + (y_dg - y_x) ** 2 + (z_dg - z_x) ** 2)

            # 构造系数矩阵
            Ax[i, j] = -G * 10.25 * s * (x_dg - x_x) / lij ** 3
            Ay[i, j] = -G * 10.25 * s * (y_dg - y_x) / lij ** 3
            Az[i, j] = -G * 10.25 * s * (z_dg - z_x) / lij ** 3

    A = np.vstack((Ax, Ay, Az))

    return A
def coeff_matrix1(Lon_dg, Lat_dg, Lon_x, Lat_x, reso):
    # 初始化参数
    h = 480e3  # m
    R = 6378136.3
    r = R + h
    G = 6.67259e-11
    s = 510067866e6 / reso  # reso为地球表面格网个数

    Ax = np.zeros((len(Lon_dg), len(Lon_x)))
    Ay = np.zeros((len(Lon_dg), len(Lon_x)))
    Az = np.zeros((len(Lon_dg), len(Lon_x)))

    for i in range(len(Lon_dg)):
        # 质量块的直角三维坐标
        x_x = R * np.cos(Lat_x) * np.cos(Lon_x)
        y_x = R * np.cos(Lat_x) * np.sin(Lon_x)
        z_x = R * np.sin(Lat_x)

        # 空间点的直角三维坐标
        x_dg = r * np.cos(Lat_dg[i]) * np.cos(Lon_dg[i])
        y_dg = r * np.cos(Lat_dg[i]) * np.sin(Lon_dg[i])
        z_dg = r * np.sin(Lat_dg[i])

        # 计算空间点和质量块之间的距离
        lij = np.sqrt((x_dg - x_x) ** 2 + (y_dg - y_x) ** 2 + (z_dg - z_x) ** 2)

        # 构造系数矩阵
        Ax[i, :] = -G * 10.25 * s * (x_dg - x_x) / lij ** 3
        Ay[i, :] = -G * 10.25 * s * (y_dg - y_x) / lij ** 3
        Az[i, :] = -G * 10.25 * s * (z_dg - z_x) / lij ** 3


    A = np.vstack((Ax, Ay, Az))

    return A
def coeff_matrix_dg(lon_dg, lat_dg, lon_x, lat_x, reso):
    R = 6378136.3
    h = 480e+3
    G = 6.67259e-11
    s = 510067866e+6 / reso
    f = 1 / 298.2572
    e2 = 2 * f - f ** 2
    A = np.zeros((lon_dg.shape[0], lon_x.shape[0]), dtype=np.float32)
    lat_x = np.squeeze(lat_x)
    lon_x = np.squeeze(lon_x)

    for i in range(lon_dg.shape[0]):
        a = R * (1 - f) / np.sqrt(1 - e2 * (np.cos(lat_x)) ** 2)
        r = a + h
        cospsi = np.sin(lat_x) * np.sin(lat_dg[i]) + np.cos(lat_dg[i]) * np.cos(lat_x) * np.cos(lon_dg[i] - lon_x)
        A[i, :] = G * 10.25 * s * (r - a * cospsi) / (a ** 2 + r ** 2 - 2 * a * r * cospsi) ** 1.5
    print('重力扰动位的系数矩阵构造完成')
    return A

def coefficient_matrix(lon_x, lat_x, lmax):
    nblock = lat_x.shape[0]
    a = 6378136.3
    gm = 3.986004415e14
    G = 6.67259e-11
    earth_mass = gm / G
    s = 4 * np.pi / nblock
    maxdegree = lmax
    # 将度转换为弧度
    lon = np.deg2rad(lon_x)
    lat = np.deg2rad(lat_x)
    longitude = np.mat(list(lon.flatten()))
    latitude = np.mat(list(lat.flatten()))
    # 勒夫负荷系数
    l = np.arange(0, maxdegree + 1, 1)
    kl = lovenums(l)
    # 计算三角函数值
    m = np.arange(0, maxdegree + 1)[:, np.newaxis]
    ccos = np.cos(np.multiply(m, longitude))
    ssin = np.sin(np.multiply(m, longitude))

    # 计算勒让德函数
    lat = list(latitude.flatten())
    Pnm0, _ = plm_holmes(maxdegree, np.sin(lat))
    costant1 = np.zeros(maxdegree + 1)
    for i in range(maxdegree + 1):
        costant1[i] = 10.25 * (1 + kl[i]) * a ** 2 / (earth_mass * (2 * i + 1)) * s

    col = (maxdegree + 2) * (maxdegree + 1)
    coe = np.zeros((col, latitude.shape[1]))
    k = 0
    for i in range(maxdegree + 1):
        for j in range(i + 1):
            if k < col / 2:
                coe[int(i * (i + 1) / 2 + j), :] = costant1[i] * np.multiply(Pnm0[i, j, :], ccos[j, :])
                coe[int(i * (i + 1) / 2 + j + col / 2), :] = costant1[i] * np.multiply(Pnm0[i, j, :], ssin[j, :])
            k += 1

    return coe
