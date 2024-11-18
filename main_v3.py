"""
Macson 拟合法-三维加速度点质量（ref:苏勇（2019）反演地表质量变化的附有空间约束的三维加速度点质量模型法)
"""
# from acceleration import acceleration
from calculate_ewh_v3 import tikhonov_lcuve
from loaddata import loadacc
import netCDF4 as nc

# 设置边界序号“0-75”，如果为0则输出全球数据
basin_id = 0
bln_path = r'D:\ill_pose\CSRMASCON\data\bln'
out_acc_path = r'D:\ill_pose\CSRMASCON\outdata\dg'
out_ewh_path = r'D:\ill_pose\CSRMASCON\outdata\ewh_v3'
# # 计算三维加速度
# acc = acceleration(r'D:\ill_pose\CSRMASCON\data\CSR_CSR-Release-06_60x60_unfiltered',
#                         r'D:\ill_pose\CSRMASCON\data\Correct_degree1_2\TN-13_GEOC_CSR_RL06.txt',
#                         r'D:\ill_pose\CSRMASCON\data\Correct_degree1_2\TN-14_C30_C20_GSFC_SLR.txt',
#                         1,
#                    bln=bln_path,
#                    index=basin_id,
#                    output_path=out_acc_path)
# 读取三维加速度及经纬度
acc, lon, lat = loadacc(out_acc_path)
# 正则化矩阵
# 正则化矩阵
fname = r'/home/master/shujw/CSRMASCON/outdata/rms/RM_PM.nc'
dataset = nc.Dataset(fname)
RM = dataset.variables["RM"][:]
# 由三维加速度计算地面质量变化/等效水高
ewh = tikhonov_lcuve(acc, lon, lat, out_ewh_path, bln_path, basin_id, rm=RM)