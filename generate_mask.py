import os
import sys

from shapely.geometry import Polygon, Point
import xarray as xr
import geopandas as gpd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def generate_mask_RMS(Antarctica_path, line_shp_file, polygon_shp_file, location_file, output_path):
    print('掩膜数据开始生成')
    # 形成南极区域的多边形
    coords = []
    with open(Antarctica_path, "r") as f:
        for i in range(1):
            f.readline()  # 跳过第一行
        for line in f:
            words = line.split()
            longitude = float(words[0].strip("'"))
            latitude = float(words[1].strip("'"))
            coord = [longitude, latitude]
            coords.append(np.array(coord))
    f.close()
    coords = np.array(coords)

    # 判断bln文件中经度范围为（0，360）还是（-180，180）
    if np.max(coords[:, 0]) > 180:  # 如果经度是0-360
        indices = np.where(coords[:, 0] > 180)
        coords[indices, 0] -= 360  # 将0-360转换为-180-180
    else:
        pass
    poly = Polygon(coords)
    # 读取shp文件
    coastline_gdf = gpd.read_file(line_shp_file)
    area_gdf = gpd.read_file((polygon_shp_file))
    # 读取网格地理位置
    grid_location = xr.open_dataset(location_file)
    lats = grid_location['lat'].values
    lons = grid_location['lon'].values
    # 获取第一个要素的几何数据
    geometries = coastline_gdf['geometry'].tolist()
    area_buffers = area_gdf['geometry'].tolist()
    # 创建地理多边形
    # 初始化一个空的Polygon对象
    poly_output = Polygon()
    for geometry in geometries:
        if geometry.geom_type == 'LineString':
            if len(geometry.coords) < 4:
                continue
            else:
                ls_poly = Polygon(geometry)
                ls_poly = ls_poly.buffer(0)
                poly_output = poly_output.union(ls_poly)

        elif geometry.geom_type == 'MultiLineString':
            for line in geometry.geoms:
                mls_poly = Polygon(line)
                poly_output = poly_output.union(mls_poly)
    # 将值赋予地理多边形
    geo_polygon = poly_output
    for area_buffer in area_buffers:
        geo_polygon = geo_polygon.union(area_buffer)

    # 制作掩膜
    num_point = len(lats)
    mask = np.zeros((num_point, 1))
    for i in range(num_point):
        lon = lons[i]
        lat = lats[i]
        if lon == 180 and lat < -80:
            mask[i] = 1
            continue
        pt = Point(lon, lat)
        if geo_polygon.contains(pt) or poly.contains(pt):
            mask[i] = 1
        else:
            mask[i] = 0
    # 存储掩膜
    ncfile = Dataset(os.path.join(output_path, 'RMS_mask.nc'), mode='w', format='NETCDF4')
    ncfile.createDimension('point', num_point)
    latitude = ncfile.createVariable('lat', np.float32, ('point',))
    longitude = ncfile.createVariable('lon', np.float32, ('point',))
    RMS_mask = ncfile.createVariable('RMS_mask', np.float32, ('point',))
    # 存储变量
    RMS_mask[:] = mask
    latitude[:] = lats
    longitude[:] = lons

    ncfile.close()

    print(f'RMS掩膜数据已经存储到{output_path}')


def generate_region_mask(region_path, lons, lats,extend=None):
    print('掩膜数据开始生成')
    # 形成南极区域的多边形
    coords = []
    with open(region_path, "r") as f:
        for i in range(1):
            f.readline()  # 跳过第一行
        for line in f:
            words = line.split()
            longitude = float(words[0].strip("'"))
            latitude = float(words[1].strip("'"))
            coord = [longitude, latitude]
            coords.append(np.array(coord))
    f.close()
    coords = np.array(coords)

    # 判断bln文件中经度范围为（0，360）还是（-180，180）
    if np.max(coords[:, 0]) > 180:  # 如果经度是0-360
        indices = np.where(coords[:, 0] > 180)
        coords[indices, 0] -= 360  # 将0-360转换为-180-180
    else:
        pass
    poly = Polygon(coords)
    if extend ==None:
        buffer_polygon = poly
    else:
        buffer_polygon = poly.buffer(extend)

    # # 可视化原始多边形和缓冲区
    # fig, ax = plt.subplots()
    # x, y = poly.exterior.xy
    # ax.plot(x, y, 'b', label='Original Polygon')
    # x, y = buffer_polygon.exterior.xy
    # ax.plot(x, y, 'r', label='Buffered Polygon')
    #
    # ax.legend()
    # plt.show()


    # 制作掩膜
    num_point = len(lats)
    mask = np.zeros((num_point, 1))
    t=0
    for i in range(num_point):
        lon = lons[i]
        lat = lats[i]
        pt = Point(lon, lat)
        if buffer_polygon.contains(pt):
            t+=1
            mask[i] = 1
        else:
            mask[i] = 0
    return mask


def gen_region_mask(filename, res=1):
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
    if np.max(coords[:, 0]) > 180:  # 如果经度是0-360
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
    mask = np.roll(mask, steps, axis=1)
    return mask


def region_select(bln_path, global_lat, global_lon):
    coords = []
    with open(bln_path, "r") as f:
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
    if np.max(coords[:, 0]) > 180:  # 如果经度是0-360
        indices = np.where(coords[:, 0] > 180)
        coords[indices, 0] -= 360  # 将0-360转换为-180-180
    else:
        pass
    poly = Polygon(coords)
    # 筛选出指定流域中的网格经纬度
    lons, lats = [], []
    for lon, lat in zip(global_lon, global_lat):
        pt = Point(lon, lat)
        if poly.contains(pt):
            lons.append(lon)
            lats.append(lat)
    lons = np.array(lons)
    lats = np.array(lats)
    return lons, lats
