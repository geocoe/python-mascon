import numpy as np

from scipy.io import loadmat
from netCDF4 import Dataset
import numpy as np
# 替换为你的MATLAB文件路径
mat_file_path = r'D:\code\point mass\data\Location.mat'

# 加载MATLAB文件
mat_data = loadmat(mat_file_path)

# MATLAB文件中的变量可以通过字典键访问
lon = mat_data['lon']
lat = mat_data['lat']

# 存储数据为netCDF格式
location_storage_path=r'D:\code\point mass\data\Grid_10242.nc'
ncfile=Dataset(location_storage_path,mode='w',format='NETCDF4')
ncfile.createDimension('number_parameter',len(lon))

latitude=ncfile.createVariable('lat',np.float32,('number_parameter',))
longitude=ncfile.createVariable('lon',np.float32,('number_parameter',))

latitude[:]=lat
longitude[:]=lon

ncfile.close()

print(f'网格位置已经存储到:{location_storage_path}')