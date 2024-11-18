import os.path
from grace_time import date2doy
from generate_mask import generate_region_mask, gen_region_mask
from result_visualization import mEWH_series_plot
import sys
from read_mascon import read_CSR_mascon
from coefficient_set import coeff_matrix_dg
import xarray as xr
import numpy as np
from l_ribbon1 import l_ribbon, l_curve
from min_mse import minMSE_IT, minMSE_NIT


def set_resolution(res):
    if res == 1:
        north, south, east, west = 89.5, -89.5, 359.5, 0.5
    elif res == 0.5:
        north, south, east, west = 89.75, -89.75, 359.75, 0.25
    elif res == 0.25:
        north, south, east, west = 89.875, -89.875, 359.875, 0.125
    else:
        sys.exit('未定义的分辨率')
    lat = np.arange(north, south - res, -res)
    lon = np.arange(west, east + res, res)
    return lon, lat


def comparision(x, a, time):
    region_path = r'D:\code\point mass\data\bln\Greenland.bln'
    location_path = r'D:\code\point mass\data\Grid_10242.nc'
    CSR_mascon_path = r'D:\code\point mass\data\mascon\CSR_GRACE_GRACE-FO_RL0602_Mascons_GSU-component.nc'

    region_name = os.path.basename(region_path)
    region_name = os.path.splitext(region_name)[0]
    # 1.mascon时间序列
    time_series = np.mean(x, axis=1)

    # 2.读取CSR mascon
    CSR_mascon, CSR_date = read_CSR_mascon(CSR_mascon_path)
    CSR_time = date2doy(CSR_date)

    # 制作掩膜生成CSR mascon的时间序列
    CSR_mask = gen_region_mask(region_path, 0.25)
    CSR_timeseries = np.sum(CSR_mascon * CSR_mask, axis=(1, 2)) / np.sum(CSR_mask)

    # 可视化时间序列
    imgpath = r'D:\code\point mass\images\Greenland_weighted\RMS_ribbon'
    imgpath = os.path.join(imgpath, f'disturb_gravity_{region_name}_timeseries_{a}_unweighted_ribbon')
    mEWH_series_plot([time_series, CSR_timeseries], [f'mascon_fit_{a}', "CSR"], [time, CSR_time], imgpath)
    print()


if __name__ == "__main__":
    shc_path = r'D:\code\point mass\data\CSR_CSR-Release-06_60x60_unfiltered'
    disgravtiy_path = r'D:\code\point mass\results\acceleration\Greenland_distrub_gravity.nc'
    mascon_path = r'D:\code\point mass\results\acc_mascon\all_TWSA_RMS_Greenland.nc'
    image_path = r'D:\code\point mass\images\Greenland_weighted'
    Rmatix_path = r'D:\code\point mass\results\Rmatrix\Rmatrix.nc'
    mask_path = r'D:\code\point mass\results\Greenland_mask\Greenland_mask.nc'

    mask_dataset = xr.open_dataset(mask_path)
    mask = mask_dataset['mask'].values

    Rmatrix_dataset = xr.open_dataset(Rmatix_path)
    Rmatrix = Rmatrix_dataset['Rmatrix'].values

    # 选择指定区域内的RMS
    R = Rmatrix[mask == 1]

    mascon_dataset = xr.open_dataset(mascon_path)
    lon_g = mascon_dataset['lon'].values
    lat_g = mascon_dataset['lat'].values
    disgravtiy_dataset = xr.open_dataset(disgravtiy_path)
    lon_d = disgravtiy_dataset['lon'].values
    lat_d = disgravtiy_dataset['lat'].values
    disgravity = disgravtiy_dataset['acc'].values
    model_name = disgravtiy_dataset['model_name'].values
    time = disgravtiy_dataset['time'].values
    # 构建重力扰动位的系数矩阵
    lon_rad_g = np.deg2rad(lon_g)
    lat_rad_g = np.deg2rad(lat_g)
    lon_rad_d = np.deg2rad(lon_d)
    lat_rad_d = np.deg2rad(lat_d)
    A = coeff_matrix_dg(lon_rad_d, lat_rad_d, lon_rad_g, lat_rad_g, 40962)
    # 加入正则化矩阵后重新构建系数矩阵
    R_S = np.diag(1 / np.sqrt(R))
    A_S = A @ R_S
    Nx = A_S.T @ A_S
    y_all = disgravity[2, :, :]
    x_all = []
    t = 0
    for y in y_all:
        num_obs = len(y)
        t = t + 1
        alpha = np.linspace(1e-15, 1e-13, 200)
        _,_,index = l_ribbon(A_S, y,15, alpha, t)
        a = alpha[index]
        b = A_S.T @ y
        NR = np.eye(len(lat_g)) * a
        N = Nx + NR
        N_inv = np.linalg.inv(N)
        x_S = N_inv @ b
        v = A_S @ x_S - y
        sigma = v.T @ v / (num_obs - np.trace(Nx @ N_inv))
        a_min = minMSE_NIT(A_S, sigma, x_S, 1e-19)
        print(a_min)
        NR_min = np.eye(len(lat_g)) * a_min
        x_S_min = np.linalg.inv(Nx + NR_min) @ b

        x = R_S @ x_S_min

        x_all.append(x)
    x_all = np.array(x_all)
    comparision(x_all, a_min, time)
    print()
