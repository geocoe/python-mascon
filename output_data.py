import os
import numpy as np
from netCDF4 import Dataset


def out_mask(output_path, mask, lons, lats):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num_point = len(lats)
    # 存储掩膜
    ncfile = Dataset(os.path.join(output_path, 'Greenland_mask.nc'), mode='w', format='NETCDF4')
    ncfile.createDimension('point', num_point)
    latitude = ncfile.createVariable('lat', np.float32, ('point',))
    longitude = ncfile.createVariable('lon', np.float32, ('point',))
    region_mask = ncfile.createVariable('mask', np.float32, ('point',))
    # 存储变量
    region_mask[:] = mask
    latitude[:] = lats
    longitude[:] = lons
    ncfile.close()
    print(f'格陵兰岛区域的掩膜已经存储到{output_path}')


def output_mascon(TWSA, W, time, lon, lat, TWSA_path):
    # 判断路径是否存在
    dir_path = os.path.dirname(TWSA_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not TWSA_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        if np.ndim(TWSA) == 2:
            (num_month, num_unknown) = np.shape(TWSA)
            num_model = W.shape[-1]
            # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
            ncfile = Dataset(TWSA_path, mode='w', format='NETCDF4')
            # 创建变量维度
            ncfile.createDimension('time', num_month)
            ncfile.createDimension('unknown', num_unknown)
            ncfile.createDimension('model', num_model)
            # 创建变量
            TWSA_data = ncfile.createVariable('lwe_thickness', np.float32, ('time', 'unknown'))
            W_data = ncfile.createVariable('variance', np.float32, ('time', 'model'))
            time_data = ncfile.createVariable('time', np.float32, ('time'))
            longitude = ncfile.createVariable('lon', np.float32, ('unknown',))
            latitude = ncfile.createVariable('lat', np.float32, ('unknown',))
            # 设置标题
            ncfile.title = 'Terrestrial Water Thickness Anomaly'
            # 填充数据
            TWSA_data[:, :] = TWSA
            W_data[:, :] = W
            time_data[:] = time
            longitude[:] = lon
            latitude[:] = lat
            # 写入完成，关闭文件
            ncfile.close()
        elif np.ndim(TWSA) == 3:
            (num_month, num_alpha, num_unknown) = np.shape(TWSA)
            num_model = W.shape[-1]
            # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
            ncfile = Dataset(TWSA_path, mode='w', format='NETCDF4')
            # 创建变量维度
            ncfile.createDimension('time', num_month)
            ncfile.createDimension('unknown', num_unknown)
            ncfile.createDimension('model', num_model)
            ncfile.createDimension('alpha', num_alpha)
            # 创建变量
            TWSA_data = ncfile.createVariable('lwe_thickness', np.float32, ('time', 'alpha', 'unknown'))
            W_data = ncfile.createVariable('variance', np.float32, ('time', 'alpha', 'model'))
            time_data = ncfile.createVariable('time', np.float32, ('time',))
            longitude = ncfile.createVariable('lon', np.float32, ('unknown',))
            latitude = ncfile.createVariable('lat', np.float32, ('unknown',))
            # 设置标题
            ncfile.title = 'Terrestrial Water Thickness Anomaly'
            # 填充数据
            TWSA_data[:, :, :] = TWSA
            W_data[:, :, :] = W
            time_data[:] = time
            longitude[:] = lon
            latitude[:] = lat
            # 写入完成，关闭文件
            ncfile.close()

        print(f"等效水高结果已保存到{TWSA_path}")


def output_TWSA(TWSA, time, TWSA_path):
    (num_month, num_unknown) = np.shape(TWSA)
    dir_path = os.path.dirname(TWSA_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not TWSA_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(TWSA_path, mode='w', format='NETCDF4')
        # 创建变量维度
        ncfile.createDimension('time', num_month)
        ncfile.createDimension('unknown', num_unknown)
        # 创建变量
        TWSA_data = ncfile.createVariable('lwe_thickness', np.float32, ('time', 'unknown'))
        time_data = ncfile.createVariable('time', np.float32, ('time',))
        # 设置标题
        ncfile.title = 'Terrestrial Water Thickness Anomaly'
        # 填充数据
        TWSA_data[:, :] = TWSA
        time_data[:] = time
        # 写入完成，关闭文件
        ncfile.close()

        print(f"等效水高结果已保存到{TWSA_path}")


def output_Rmatrix(R, lon, lat, Rmatrix_path):
    num_unknown = len(R)
    dir_path = os.path.dirname(Rmatrix_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not Rmatrix_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(Rmatrix_path, mode='w', format='NETCDF4')
        # 创建变量维度
        ncfile.createDimension('unknown', num_unknown)
        # 创建变量
        R_data = ncfile.createVariable('Rmatrix', np.float32, ('unknown',))
        longitude = ncfile.createVariable('lon', np.float32, ('unknown',))
        latitude = ncfile.createVariable('lat', np.float32, ('unknown',))
        # 设置标题
        ncfile.title = 'regularization matrix'
        # 填充数据
        R_data[:] = R
        longitude[:] = lon
        latitude[:] = lat
        # 写入完成，关闭文件
        ncfile.close()

        print(f"正则化矩阵已保存到{Rmatrix_path}")


def out_acceleration(acc, time, long, lat, path, bln_name, model_name):
    nc_name = f'{bln_name}_distrub_gravity.nc'
    # 判断指定文件夹是否存在，若不存在则重新创建
    if not os.path.exists(path):
        os.makedirs(path)
    output_path = os.path.join(path, nc_name)
    if not output_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        num_model, num_month, num_obs = np.shape(acc)
        num_grid = len(lat)
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(output_path, mode='w', format='NETCDF4')

        # 创建一个维度
        ncfile.createDimension('time', num_month)
        ncfile.createDimension('observation', num_obs)
        ncfile.createDimension('model', num_model)
        ncfile.createDimension('grid', num_grid)

        ncfile.createDimension('string_dim', None)

        # 创建一个变量来存储字符串数据
        string_var = ncfile.createVariable('model_name', str, ('string_dim',))

        DG = ncfile.createVariable('acc', np.float32, ('model', 'time', 'observation'))
        times = ncfile.createVariable('time', np.float32, ('time',))
        longitude = ncfile.createVariable('lon', np.float32, ('grid',))
        latitude = ncfile.createVariable('lat', np.float32, ('grid',))
        # 设置标题
        ncfile.title = 'acceleration in three different directions'

        DG.units = 'ugal'
        times.units = 'decimal year'

        DG[:, :, :] = acc
        times[:] = time
        longitude[:] = long[:]
        latitude[:] = lat[:]

        # 将字符串数据写入变量
        string_var[:] = np.array(model_name, dtype='object')

        # 关闭nc文件
        ncfile.close()
        print(f"重力扰动位结果已保存为{output_path}文件")
