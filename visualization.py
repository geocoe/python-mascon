import os
import sys
import json
import time
from utils import path_check, select_from_tempValue
from grace_time import time_elapsed_show
from result_visualization import gmt_show_mascon, check_mascon, show_mask, show_mask_Antarctica, show_Rmatrix, \
    outputGif, show_Greenland
from grace_time import doy2date
from utils import check_ncfile

if __name__ == "__main__":
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
        josn_data = f.read().strip()
    # 解析json文件
    point_mass = json.loads(josn_data)

    # 输入路径
    grid_location_path = point_mass["Inputfilepath"][1]["grid_location_path"]

    # 输出路径
    mascon_path = point_mass["Outputfilepath"][0]["mascon_path"]
    image_path = point_mass["Outputfilepath"][1]["image_path"]
    RMS_mask_path = point_mass["Outputfilepath"][2]["RMS_mask_path"]
    RMS_mask_image_path = point_mass["Outputfilepath"][3]["RMS_mask_image_path"]
    TWSA_path = point_mass["Outputfilepath"][4]["TWSA_path"]
    Rmatrix_path = point_mass["Outputfilepath"][5]["Rmatrix_path"]
    Rmatrix_image_path = point_mass["Outputfilepath"][6]["Rmatrix_image_path"]
    TWSA_image_path = point_mass["Outputfilepath"][7]["TWSA_image_path"]
    mascon_gif_path = point_mass["Outputfilepath"][8]["mascon_gif_path"]

    # 检查文件路径并且对文件路径进行转换
    grid_location_path = path_check(grid_location_path)
    mascon_path = path_check(mascon_path)
    image_path = path_check(image_path)
    RMS_mask_path = path_check(RMS_mask_path)
    RMS_mask_image_path = path_check(RMS_mask_image_path)
    TWSA_path = path_check(TWSA_path)
    Rmatrix_path = path_check(Rmatrix_path)
    Rmatrix_image_path = path_check(Rmatrix_image_path)
    TWSA_image_path = path_check(TWSA_image_path)
    mascon_gif_path = path_check(mascon_gif_path)

    # 程序入口筛选
    Console = point_mass["Console"][0]
    entrance_choice = select_from_tempValue(Console)

    # 程序执行入口

    if entrance_choice == 'show_mascon':
        check_mascon(mascon_path, grid_location_path, image_path)

    elif entrance_choice == 'show_mask':
        if os.path.exists(RMS_mask_path):
            file = os.listdir(RMS_mask_path)
            file_path = os.path.join(RMS_mask_path, file[0])
            show_mask(file_path, RMS_mask_image_path, "RMS_10242")
            show_mask_Antarctica(file_path, RMS_mask_image_path, "southpolarRMS_10242")
    elif entrance_choice == 'show_Rmatrix':
        if os.path.exists(Rmatrix_path):
            file = os.listdir(Rmatrix_path)
            file_path = os.path.join(Rmatrix_path, file[0])
            show_Rmatrix(file_path, Rmatrix_image_path, "Rmatrix")
    elif entrance_choice == 'show_TWSA':
        check_mascon(TWSA_path, grid_location_path, TWSA_image_path)
    elif entrance_choice == 'show_gif':
        outputGif(image_path, mascon_gif_path)
    elif entrance_choice == 'Greenland':
        mascon_dataset = check_ncfile(mascon_path)

        show_Greenland(mascon_dataset,image_path)

    # 程序结束时间
    end_time_procedure = time.time()
    # 显示程序运行时间
    time_elapsed_show(start_time_procedure, end_time_procedure)
