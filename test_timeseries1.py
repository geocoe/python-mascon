import os.path
from grace_time import date2doy
import xarray as xr
from generate_mask import generate_region_mask,gen_region_mask
import numpy as np
from result_visualization import mEWH_series_plot
import sys
from read_mascon import read_CSR_mascon

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

if __name__=='__main__':
    #
    mascon_path = r'D:\code\point mass\results\acc_mascon\all_TWSA_RMS_Greenland.nc'
    region_path = r'D:\code\point mass\data\bln\Greenland.bln'
    location_path = r'D:\code\point mass\data\Grid_10242.nc'
    CSR_mascon_path = r'D:\code\point mass\data\mascon\CSR_GRACE_GRACE-FO_RL0602_Mascons_GSU-component.nc'

    region_name = os.path.basename(region_path)
    region_name = os.path.splitext(region_name)[0]


    # 制作掩膜并生成时间序列
    mascon_datasset= xr.open_dataset(mascon_path)
    mascon_data = mascon_datasset['lwe_thickness'].values
    time = mascon_datasset['time'].values
    # mascon_data1 = mascon_data[:,10,:]
    time_series = np.mean(mascon_data, axis=1)

    # 2.读取CSR mascon
    CSR_mascon, CSR_date = read_CSR_mascon(CSR_mascon_path)
    CSR_time = date2doy(CSR_date)

    # 制作掩膜生成CSR mascon的时间序列
    CSR_mask = gen_region_mask(region_path,0.25)
    CSR_timeseries = np.sum(CSR_mascon * CSR_mask,axis=(1,2))/np.sum(CSR_mask)


    # 可视化时间序列
    imgpath = r'D:\code\point mass\images'
    imgpath = os.path.join(imgpath,f'disturb_gravity_{region_name}_timeseries')
    mEWH_series_plot([time_series,CSR_timeseries], ['mascon_fit',"CSR"], [time,CSR_time], imgpath)
    print()



