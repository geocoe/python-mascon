import os
import sys
import json
import time
from utils import path_check, select_from_tempValue
from mascon_fit_emsenble import mascon_calculation, regularization_matrix, acceleration_mascon
from grace_time import time_elapsed_show
from generate_mask import generate_region_mask
import xarray as xr
from output_data import out_mask

if __name__ == '__main__':
    # 程序开始时间
    print('程序开始运行')
    # sys.exit('程序正常运行')
    start_time_procedure = time.time()
    # 获取可执行文件路径和目录路径
    executable_path = os.path.realpath(sys.argv[0])
    executable_dir = os.path.dirname(executable_path)
    # 拼接文件路径
    file_path = os.path.join(executable_dir, 'point_mass.json')
    # 打开json文件
    with open(file_path, encoding='utf-8') as f:
        json_data = f.read().strip()
    # 解析json文件
    point_mass = json.loads(json_data)

    # 输入路径
    SH_path = point_mass["Inputfilepath"][0]["SH_path"]
    grid_location_path = point_mass["Inputfilepath"][1]["grid_location_path"]
    R_matrix_path = point_mass["Inputfilepath"][2]["reg_matrix_path"]
    polygon_path = point_mass['Inputfilepath'][3]['polygon_shpfile_path']
    coastline_path = point_mass['Inputfilepath'][4]['coastline_shpfile_path']
    Antarctica_path = point_mass['Inputfilepath'][5]['Antarctica_path']
    DDK_filter_path = point_mass['Inputfilepath'][6]['DDK_filter_path']
    tn13_path = point_mass['Inputfilepath'][7]['tn13_path']
    tn14_path = point_mass['Inputfilepath'][8]['tn14_path']
    bln_path = point_mass['Inputfilepath'][9]['bln_path']

    # 输出路径
    mascon_path = point_mass["Outputfilepath"][0]["mascon_path"]
    RMS_mask_path = point_mass["Outputfilepath"][2]["RMS_mask_path"]
    TWSA_path = point_mass["Outputfilepath"][4]["TWSA_path"]
    Rmatrix_path = point_mass["Outputfilepath"][5]["Rmatrix_path"]
    acceleration_path = point_mass["Outputfilepath"][9]["acceleration_path"]
    Gmask_extended_path = point_mass["Outputfilepath"][10]["Gmask_extended_path"]

    # 输入参数
    number_model = point_mass['Inputparameter'][0]
    constraint_type = point_mass['Inputparameter'][1]
    extend = point_mass['Inputparameter'][2]['value']


    # 检查文件路径并且对文件路径进行转换
    SH_path = path_check(SH_path)
    grid_location_path = path_check(grid_location_path)
    mascon_path = path_check(mascon_path)
    R_matrix_path = path_check(R_matrix_path)
    number_model = select_from_tempValue(number_model)
    constraint_type = select_from_tempValue(constraint_type)
    polygon_path = path_check(polygon_path)
    coastline_path = path_check(coastline_path)
    RMS_mask_path = path_check(RMS_mask_path)
    Antarctica_path = path_check(Antarctica_path)
    DDK_filter_path = path_check(DDK_filter_path)
    TWSA_path = path_check(TWSA_path)
    Rmatrix_path = path_check(Rmatrix_path)
    acceleration_path = path_check(acceleration_path)
    tn13_path = path_check(tn13_path)
    tn14_path = path_check(tn14_path)
    bln_path = path_check(bln_path)
    Gmask_extended_path = path_check(Gmask_extended_path)
    extend = int(extend)

    # 集成文件路径
    kwarg = {'SH_path': SH_path, 'grid_location_path': grid_location_path, 'mascon_path': mascon_path,
             'model_employed': number_model, 'Rmatrix_path': Rmatrix_path, 'constraint_type': constraint_type}

    acc_kwarg = {'SH_path': SH_path, 'grid_location_path': grid_location_path, 'mascon_path': mascon_path,
                 'model_employed': number_model, 'Rmatrix_path': Rmatrix_path, "tn13_path": tn13_path,
                 "tn14_path": tn14_path, "bln_path": bln_path, 'constraint_type': constraint_type,
                 "acceleration_path": acceleration_path, "RMS_mask_path": RMS_mask_path,
                 'Gmask_extended_path': Gmask_extended_path}

    # 程序入口选择
    Console = point_mass["Console"][0]
    entrance_choice = select_from_tempValue(Console)

    # 程序执行入口
    if entrance_choice == 'masconfit':
        mascon_calculation(**kwarg)

    elif entrance_choice == 'acceleration':
        acceleration_mascon(**acc_kwarg)

    elif entrance_choice == 'generate_mask':
        # generate_mask_RMS(Antarctica_path, coastline_path, polygon_path, grid_location_path, RMS_mask_path)
        grid_dataset = xr.open_dataset(grid_location_path)
        lon = grid_dataset['lon'].values
        lat = grid_dataset['lat'].values
        mask = generate_region_mask(bln_path, lon, lat,extend)
        out_mask(Gmask_extended_path, mask, lon, lat)


    elif entrance_choice == "regularization_matrix_construction":
        regularization_matrix(SH_path, DDK_filter_path, grid_location_path, TWSA_path, RMS_mask_path, Rmatrix_path)
    # 程序结束时间
    end_time_procedure = time.time()
    # 显示程序运行时间
    time_elapsed_show(start_time_procedure, end_time_procedure)
