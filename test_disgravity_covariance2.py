
import xarray as xr
import numpy as np



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


if __name__ == "__main__":
    # 路径设置
    shc_path = r'D:\code\point mass\data\CSR_CSR-Release-06_60x60_unfiltered'
    disgravtiy_path = r'D:\code\point mass\results\acceleration\Greenland_distrub_gravity.nc'
    mascon_path = r'D:\code\point mass\results\acc_mascon\all_TWSA_RMS_Greenland.nc'
    image_path = r'D:\code\point mass\images\Greenland_weighted'
    mascon_dataset = xr.open_dataset(mascon_path)
    lon_g = mascon_dataset['lon'].values
    lat_g = mascon_dataset['lat'].values
    disgravtiy_dataset = xr.open_dataset(disgravtiy_path)
    lon_d = disgravtiy_dataset['lon'].values
    lat_d = disgravtiy_dataset['lat'].values
    disgravity = disgravtiy_dataset['acc'].values
    time = disgravtiy_dataset['time'].values
    # 构建重力扰动位的系数矩阵
    lon_rad_g = np.deg2rad(lon_g)
    lat_rad_g = np.deg2rad(lat_g)
    lon_rad_d = np.deg2rad(lon_d)
    lat_rad_d = np.deg2rad(lat_d)
    A = coeff_matrix_dg(lon_rad_d, lat_rad_d, lon_rad_g, lat_rad_g, 40962)

    Nx = A.T @ A
    y_all = disgravity[1, :, :]
    alpha = [8e-19,1e-18,8e-18]
    for a in alpha:
        NR = np.eye(len(lat_g)) * a
        N = Nx + NR
        N_inv = np.linalg.inv(N)
        x_all = []
        for y in y_all:
            b = A.T @ y
            x = N_inv @ b
            x_all.append(x)
        x_all = np.array(x_all)
        print()
