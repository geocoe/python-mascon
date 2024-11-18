import sys
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from read_gsm import read_all_gsm
from gsm_deaverage import remove_baseline
from replace_SHs import replace_shc
import DisturbGravity
import os
from os import path, walk
import coords_swap
from netCDF4 import Dataset
import netCDF4 as nc


def select_region_from_file(basin_id=76, folder_path=None):
    # 从内置的流域中选择研究区域，流域编号只能是数字
    # basin_id取值范围1-75
    # 暂未使用
    filelist = []
    if folder_path is None:
        sys.exit('未指定边界文件路径，程序退出')
    if not os.path.exists(folder_path):
        sys.exit('边界文件路径不存在，程序退出')
    if not os.listdir(folder_path):
        sys.exit("文件夹无边界文件，程序退出")

    for (dirname, dirs, files) in walk(folder_path):
        pass
    # Sort files by name sequences.
    files = np.sort(files)
    for filename in files:
        if filename.endswith('.bln'):
            filelist.append(path.join(dirname, filename))

    # 若输入的编号非数值，则程序退出
    while True:
        try:
            basin_id = int(basin_id)
            break
        except ValueError:
            sys.exit("研究区域编号必须是1-75之间的数字，程序退出")

    if 1 <= basin_id <= 75:
        blnpath = filelist[basin_id - 1]
    else:
        sys.exit("未定义的研究区域，程序退出")

    return blnpath


# 由边界文件生成1度X1度的规则格网点
def gen_mask(filename, res=1):
    # 输入：filename流域边界经纬度文件的路径,res掩膜的空间分辨率。缺省值为1度
    # 输出：掩膜(0-360)
    # 所生成掩膜为纬度90N-90S，经度0-360
    coords = []
    with open(filename, "r") as f:
        for i in range(1):
            f.readline()  # 跳过第一行
        for line in f:
            words = line.split()
            lon = float(words[0].strip("'"))
            lat = float(words[1].strip("'"))
            coord = [lon, lat]
            coords.append(np.array(coord))
    f.close()
    coords = np.array(coords)

    # 判断bln文件中经度范围为（0，360）还是（-180，180）
    if np.max(coords[:, 0]) > 180:  # 0-360
        indices = np.where(coords[:, 0] > 180)
        coords[indices, 0] -= 360  # 将0-360转换为-180-180
    else:
        pass
    poly = Polygon(coords)

    if res == 1:
        north, west, steps = 89.5, -179.5, 180
        mask = np.zeros((180, 360))
    elif res == 0.5:
        north, west, steps = 89.75, -179.75, 360
        mask = np.zeros((360, 720))
    elif res == 0.25:
        north, west, steps = 89.875, -179.875, 720
        mask = np.zeros((720, 1440))
    else:
        sys.exit('未定义的分辨率')

    for i in range(steps):
        for j in range(2 * steps):
            lats = north - i * res
            lons = west + j * res

            pt = Point(lons, lats)  # 判断点是否在区域内，在区域内掩膜为1，否则为0
            if poly.contains(pt):
                mask[i, j] = 1
            else:
                mask[i, j] = 0

    mask = coords_swap.longitude_swap(mask)  # 将经度由-180-180转换为0-360
    # mask = np.where(mask, mask, np.nan)   #考虑是否将为0处的矩阵元素变为Nan
    return mask


# 根据边界文件生成区域格网点经纬度
def read_region_bln(bln, index, reso):
    filename = select_region_from_file(index, bln)
    mask = gen_mask(filename, reso)
    latitude, longitude = np.nonzero(mask)
    lon0 = np.mat(longitude).T
    lat0 = np.mat(latitude).T
    lon1 = lon0.copy()
    for i in range(lon1.shape[0]):
        if lon1[i] > 180:
            lon1[i] = lon1[i] - 360
    lat1 = 90 - lat0.copy()
    region = np.column_stack((lon0, lat0))
    lon = np.deg2rad(lon1)
    lat = np.deg2rad(lat1)
    return lon, lat, region, mask


# 输出全球每个格网点经纬度
def world_bln(res=1):
    row_start = 89.5
    row_end = -89.5
    col_start = -179.5
    col_end = 179.5
    if res == 1:
        mask = np.zeros((180, 360))
        row_num = 180
        col_num = 360
        lat0 = np.linspace(row_start, row_end, row_num).reshape(-1, 1)
        lon0 = np.linspace(col_start, col_end, col_num).reshape(-1, 1)
        lat0 = np.repeat(lat0.flatten(), col_num).reshape(-1, 1)
        lon0 = np.tile(lon0, row_num).T.reshape(-1, 1)
    elif res == 0.5:
        mask = np.zeros((360, 720))
        row_num = 360
        col_num = 720
        lat0 = np.linspace(row_start, row_end, row_num).reshape(-1, 1)
        lon0 = np.linspace(col_start, col_end, col_num).reshape(-1, 1)
        lat0 = np.repeat(lat0.flatten(), col_num).reshape(-1, 1)
        lon0 = np.tile(lon0, row_num).T.reshape(-1, 1)
    elif res == 0.25:
        mask = np.zeros((720, 1440))
        row_num = 720
        col_num = 1440
        lat0 = np.linspace(row_start, row_end, row_num).reshape(-1, 1)
        lon0 = np.linspace(col_start, col_end, col_num).reshape(-1, 1)
        lat0 = np.repeat(lat0.flatten(), col_num).reshape(-1, 1)
        lon0 = np.tile(lon0, row_num).T.reshape(-1, 1)
    else:
        sys.exit('未定义的分辨率')
    lon = np.deg2rad(lon0)
    lat = np.deg2rad(lat0)
    return lon, lat, mask


# 保存输出结果为nc文件
def out_nc(dg, time, mask, long, lat, path=None, bln=None, basin_id=None):
    if basin_id != 0:
        blnname = select_region_from_file(basin_id, bln)
    else:
        blnname = None
    if blnname != None:
        dirStr, _ = os.path.splitext(blnname)
        fname = dirStr.split("\\")[-1]
        nc_name = f'{fname}_DisGravity.nc'
    else:
        nc_name = f'worldwideDisGravity1.nc'
    output_path = os.path.join(path, nc_name)
    if not output_path.endswith(".nc"):
        # 格式异常时，报错并跳过
        print('文件格式异常，只支持保存为NetCDF格式')
        pass
    else:
        # 输入文件名正常时，将全球扰动重力保存为NetCDF格式
        ncfile = Dataset(output_path, mode='w', format='NETCDF4')

        ncfile.createDimension('time', len(time))
        ncfile.createDimension('x', mask.shape[0])
        ncfile.createDimension('y', mask.shape[1])
        ncfile.createDimension('long', long.shape[0])
        ncfile.createDimension('lats', lat.shape[0])


        DG = ncfile.createVariable('disgravity', np.float32, ('time', 'long'))
        times = ncfile.createVariable('time', np.float32, ('time',))
        masks = ncfile.createVariable('mask', np.float32, ('x', 'y'))
        longitude = ncfile.createVariable('long', np.float32, 'long')
        latitude = ncfile.createVariable('lat', np.float32, 'lats')
        # 设置标题
        ncfile.title = 'disturb_gravity'

        DG.units = 'ugal'
        times.units = 'decimal year'

        DG[:, :] = dg
        times[:] = time
        masks[:, :] = mask
        longitude[:] = long[:, 0]
        latitude[:] = lat[:, 0]
        ncfile.close()

        print(f"扰动重力结果已保存为{output_path}文件")


# 计算区域重力扰动/默认全球
def calculate_dg(gsm_path, tn13, tn14, reso=1, bln=None, index=None, output_path=None):
    # 计算选定区域扰动重力
    # 设定TWSA分辨率，目前可选0.25，0.5，和1度
    if index == 0:
        lon, lat, mask = world_bln(reso)
        dataset = nc.Dataset(r'D:\ill_pose\CSRMASCON\data\Lon_Lat_vec.nc')
        lon = np.deg2rad(dataset.variables["lon"][:])
        lat = np.deg2rad(dataset.variables["lat"][:])
    else:
        lon, lat, region, mask = read_region_bln(bln, index, reso)
    # 读取GSM文件
    cs, _, time, _, lmax, r, gm = read_all_gsm(gsm_path)

    # 进行C10，C11，S11，C20，C30替换
    cs_replace = replace_shc(cs, time, tn13, tn14)

    # 扣除2004年至2009年的平均值（2004-2009）
    cs_anomy = remove_baseline(cs_replace, time, 2004, 2010)

    # GIA改正

    gravitydis = []
    for clm in tqdm(cs_anomy, desc='扰动重力计算中'):  #[100:101,:,:,:]
        # 分离C、S系数
        cnm = clm[0, :, :]
        snm = clm[1, :, :]
        # 计算重力扰动
        gradis = DisturbGravity.disturbgravity_shu(lon, lat, cnm, snm, r, gm)
        gravitydis.append(gradis)

    disturb_gravity = np.array(gravitydis)[:, :, 0]
    # 将扰动重力数据赋值给掩膜矩阵
    mask = np.where(mask, mask, np.nan)
    # dg = np.repeat(mask[:, :, np.newaxis], time.shape[0], axis=2)

    out_nc(disturb_gravity, time, mask, lon, lat, output_path, bln, index)
    return disturb_gravity, time
    # return dg, time
