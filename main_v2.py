# Macson 拟合法-直接计算从球谐系数到等效水高变化（ref:谷延超（2018）顾及先验信息的时变重力场模型信号提取及其负荷形变应用)
from calculate_ewh_v2 import ewh_v2
import netCDF4 as nc

# 设置边界序号“0-75”，如果为0则输出全球数据
basin_id = 0
bln_path = '/home/master/shujw/CSRMASCON/data/bln'
out_dg_path = '/home/master/shujw/CSRMASCON/outdata/dg'
out_ewh_path = '/home/master/QK/pointmass/results'

# 正则化矩阵
fname = r'/home/master/shujw/CSRMASCON/outdata/rms/RM_PM.nc'
dataset = nc.Dataset(fname)
RM = dataset.variables["RM"][:]
#RM = None

'''
计算区域等效水高
输入：
GSM文件文件夹
TN13文件路径
TN14文件路径
分辨率
边界文件路径
区域序号:索引为0，则输出全球扰动重力
保存路径
输出：扰动重力，历元,区域
'''
ewh, time = ewh_v2(r'/home/master/shujw/CSRMASCON/data/CSR_CSR-Release-06_60x60_unfiltered_158',
                        r'/home/master/shujw/CSRMASCON/data/Correct_degree1_2/TN-13_GEOC_CSR_RL06.txt',
                        r'/home/master/shujw/CSRMASCON/data/Correct_degree1_2/TN-14_C30_C20_GSFC_SLR.txt',
                        1,
                        bln_path,
                        output_path=out_ewh_path)

