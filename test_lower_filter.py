from harmonic_summation import harmonic_summation, spherical_analysis
from coefficient_set import coeff_matrix_dg
import xarray as xr
import numpy as np
from result_visualization import figure_one

disgravity_path = r'D:\code\point mass\results\acceleration\Greenland_distrub_gravity.nc'
disgravity_dataset = xr.open_dataset(disgravity_path)
mascon_path = r'D:\code\point mass\results\acc_mascon\all_TWSA_RMS_Greenland.nc'
mascon_dataset = xr.open_dataset(mascon_path)
lon_g = mascon_dataset['lon'].values
lat_g = mascon_dataset['lat'].values
lon_d = disgravity_dataset['lon'].values
lat_d = disgravity_dataset['lat'].values
lon_rad_g = np.deg2rad(lon_g)
lat_rad_g = np.deg2rad(lat_g)
lon_rad_d = np.deg2rad(lon_d)
lat_rad_d = np.deg2rad(lat_d)
# 构建系数矩阵
A_unfiltered = coeff_matrix_dg(lon_rad_d, lat_rad_d, lon_rad_g, lat_rad_g, 40962)
num_obs, num_unknown = np.shape(A_unfiltered)
A_filtered = np.zeros((num_obs,num_unknown))
for i in range(num_unknown):
    A_colum = A_unfiltered[:, i]
    clm,slm=spherical_analysis(A_colum,lon_d,lat_d,LMAX=120,MMAX=120,resolution=1)
    clm1 = clm[0:61,0:61]
    slm1 = slm[0:61,0:61]
    A_colum_new=harmonic_summation(clm1,slm1,lon_d,lat_d)
    A_filtered[:,i]=A_colum_new
N_unfiltered = A_unfiltered.T@A_unfiltered
N_filtered = A_filtered.T@A_filtered
cond_filtered = np.linalg.cond(A_filtered)
cond_unfiltered = np.linalg.cond(A_unfiltered)
print(f'滤波前条件数为:{cond_unfiltered},滤波后条件数为:{cond_filtered}')
# 解一个月的等效水高
disgravity = disgravity_dataset['acc'].values
y=disgravity[0,0,:]
x = np.linalg.inv(N_filtered)@A_filtered.T@y
# 可视化
image_path = r'D:\code\point mass\images'
image_name = 'Greenland_filtered_A'
figure_one(x,lon_g,lat_g,image_path,image_name)