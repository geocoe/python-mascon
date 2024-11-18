from os import path, walk
from datetime import date, timedelta
import numpy as np
import os
import sys
import re
import datetime


def is_leap_year(year):
    dpm_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dpm_stnd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 4000 != 0):
        return np.array(dpm_leap, dtype=np.float64)
    else:
        return np.array(dpm_stnd, dtype=np.float64)


def get_gsm_info(filename):
    '''
    function：extract time and order information from gsm filename
    examples: gsm_info = get_gsm_info('GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc')
    '''
    start_year, start_doy = int(filename[6:10]), int(filename[10:13])
    end_year, end_doy = int(filename[14:18]), int(filename[18:21])
    sh_order = filename[33:35]
    if sh_order == "BA":
        lmax = 60
    elif sh_order == "BB":
        lmax = 96

    # Convert day of year to year, month, day
    start_date = date(start_year, 1, 1) + timedelta(days=start_doy - 1)
    start_year = int(start_date.strftime('%Y'))
    start_month = int(start_date.strftime('%m'))
    start_day = int(start_date.strftime('%d'))

    end_date = date(end_year, 1, 1) + timedelta(days=end_doy - 1)
    end_year = int(end_date.strftime('%Y'))
    end_month = int(end_date.strftime('%m'))
    end_day = int(end_date.strftime('%d'))

    dpy = is_leap_year(start_year).sum()
    dpy_end = is_leap_year(end_year).sum()
    # For data that crosses years
    end_cyclic = ((end_year - start_year) * dpy + end_doy)
    # Calculate mid-month value
    mid_day = np.mean([start_doy, end_cyclic])
    # calculate the mid-month date in decimal form
    if mid_day <= dpy:
        grace_time = start_year + mid_day / dpy
    else:
        grace_time = end_year + (mid_day - dpy) / dpy_end

    return {'start': {'year': start_year, 'month': start_month, 'day': start_day}, \
            'end': {'year': end_year, 'month': end_month, 'day': end_day}, \
            'degree': {'lmax': lmax}, \
            'time': {'mid_day': grace_time}}


def read_single_gsm(filename=None):
    '''
    function：read SHC data from a single GSM file
    examples: info, SHs, _ ,lmax = read_single_gsm('/home/user/Downloads/CSR_CSR-Release-06_60x60_unfiltered/GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc')
    '''
    if filename is None:
        sys.exit('未指定GSM文件路径，程序退出')
    if not os.path.exists(filename):
        sys.exit('GSM文件不存在，程序退出')
    if os.path.isfile(filename) and not filename.endswith('.gfc'):
        sys.exit('非.gfc文件，程序退出')

    with open(filename, 'r') as file:
        lines = file.readlines()[:2]
        string = lines[1].strip()
    if 'model converted into ICGEM-format' not in string:
        sys.exit('GSM文件格式不匹配，程序退出')

    file_name = os.path.basename(filename)
    info = get_gsm_info(file_name)
    with open(filename, 'r') as fid:
        head_index = 1
        tline = fid.readline()
        while len(tline) < 11 or tline[0:11] != 'end_of_head':
            head_index += 1
            tline = fid.readline()
            if len(tline) > 10 and tline[0:10] == 'max_degree':
                lmax = int(tline[11:])
            if len(tline) > 10 and tline[0:6] == 'radius':
                r = float(tline[7:])
            if len(tline) > 10 and tline[0:22] == 'earth_gravity_constant':
                gm = float(tline[23:])

    clm = np.zeros((2, lmax + 1, lmax + 1))
    clm_std = np.zeros_like(clm)
    clm[0, 0, 0] = 1  # C00 = 1

    with open(filename, "r") as f:
        for i in range(head_index):
            f.readline()
        for line in f:
            words = line.split()
            l, m = int(words[1]), int(words[2])

            if m > lmax: break
            if l > lmax: continue

            value_cs = [float(words[3]), float(words[4])]
            value_cs_std = [float(words[5]), float(words[6])]
            clm[:, l, m] = value_cs
            clm_std[:, l, m] = value_cs_std
    f.close()
    return info, clm, clm_std, lmax, r, gm


def read_all_gsm(folder_path=None):
    '''
    function：read all gsm files in the folder
    examples:shc, shc_std, gtime, time_cover, lmax = read_all_gsm('/home/user/Downloads/CSR_CSR-Release-06_60x60_unfiltered/')
    '''
    filelist = []
    time_cover, gtime = [], []
    shc, shc_std = [], []
    if folder_path is None:
        sys.exit('未指定GSM文件夹，程序退出')
    if not os.path.exists(folder_path):
        sys.exit('GSM文件夹不存在，程序退出')
    if not os.listdir(folder_path):
        sys.exit("文件夹无GSM文件，程序退出")
    else:
        for (dirname, dirs, files) in walk(folder_path): pass
        # Sort files by month sequences.
        files = np.sort(files)
        for filename in files:
            if 'GSM' in filename and filename.endswith('.gfc'):
                filelist.append(path.join(dirname, filename))

        for fpath in filelist:
            info, clm, clm_std, lmax, r, gm = read_single_gsm(fpath)
            shc.append(clm)
            shc_std.append(clm_std)
            doy = info['time']['mid_day']
            coverage = str(info['start']['year']) + '/' + str(info['start']['month']) + '/' + str(
                info['start']['day']) + \
                       ' to ' + str(info['end']['year']) + '/' + str(info['end']['month']) + '/' + str(
                info['end']['day'])
            time_cover.append(coverage)
            gtime.append(doy)
        shc, shc_std, gtime = np.array(shc), np.array(shc_std), np.array(gtime)
    return shc, shc_std, gtime, time_cover, lmax, r, gm


def get_file_info(filename):
    '''
    function：extract time and order information from gsm filename
    examples: gsm_info = get_gsm_info('GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc')
    '''
    # 读取AIUB
    search0, search1 = re.search('\d{4}_\d{2}.gfc', filename), re.search('90_\d{4}.gfc', filename)
    # 读取三道机构
    search2 = re.search('\d{7}', filename)
    # 读取Tongji
    search3 = re.search('\d{4}-\d{2}.gfc', filename)
    # 读取HUST
    search4 = re.search('n60-\d{6}.gfc',filename)
    if search1 or search0:
        # pdb.set_trace()
        if search1:
            search1 = search1.group()
            search1 = search1.split('_')[1]
            year, month = int('20' + search1[0:2]), int(search1[2:4])
        if search0:
            search0 = search0.group()
            search0 = search0.split('_')[0]
            year, month = int('20' + search0[0:2]), int(search0[2:4])

    elif search2:
        search2 = search2.group()
        year, day = int(search2[0:4]), int(search2[4:7])
        date_solu = date(year, 1, 1) + timedelta(day)
        month = int(date_solu.strftime('%m'))
    elif search3:
        date_solu = search3.group()
        year, month = int(date_solu[0:4]), int(date_solu[5:7])
    elif search4:
        date_solu = search4.group()
        year, month = int(date_solu[4:8]), int(date_solu[8:10])

    else:
        sys.exit('输入的时变重力场不在实验范围内')

    return f'{year}-{month:02}'


'''
info = get_gsm_info('XXXX-01_MONTHS_DQM-GAC-2_2022244-2022273_DLBA01_0001.txt')
'''


def read_single_file(filename=None, lmax=None):
    '''
    function：read SHC data from a single GSM file
    examples: info, SHs, _ ,lmax = read_single_gsm('/home/user/Downloads/CSR_CSR-Release-06_60x60_unfiltered/GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc')
    '''
    if filename is None:
        sys.exit('未指定GSM文件路径，程序退出')
    if not os.path.exists(filename):
        sys.exit('GSM文件不存在，程序退出')
    if os.path.isfile(filename) and not filename.endswith('.gfc'):
        sys.exit('非.gfc文件，程序退出')

    file_name = os.path.basename(filename)
    date = get_file_info(file_name)
    # 读取头文件
    with open(filename, 'r', encoding='utf-8') as fid:
        head_index = 1
        found_end_of_head = False
        line_count = 0
        tline = fid.readline()
        while line_count < 200 and 'end_of_head' not in tline:
            head_index += 1
            tline = fid.readline()
            # 读取物理参数
            if 'earth_gravity_constant' in tline:
                GM_str = tline.split(' ')[-1]
                GM = float(GM_str.strip())
            if 'radius' in tline:
                R_str = tline.split(' ')[-1]
                R = float(R_str.strip())
            if 'tide_system' in tline:
                tide_sys_str = tline.split(' ')[-1]
                tide_sys = tide_sys_str.strip()
            # 标记是否找到 "end_of_head"
            if 'end_of_head' in tline.strip() or 'end_of_header' in tline.strip():
                found_end_of_head = True
            if len(tline) > 10 and (tline[0:10] == 'max_degree' or tline[3:13] == 'max_degree'):
                if lmax == None:
                    lmax_str = tline.split(' ')[-1]
                    lmax = lmax_str.strip('\n')
                    lmax = int(lmax)

            line_count += 1
    fid.close()
    # 如果没有找到 "end_of_head"，抛出异常
    if not found_end_of_head:
        raise ValueError("GSM文件格式不匹配")
    clm = np.zeros((2, lmax + 1, lmax + 1))
    clm_std = np.zeros_like(clm)
    clm[0, 0, 0] = 1  # C00 = 1

    with open(filename, "r", encoding='utf-8') as f:
        for i in range(head_index):
            f.readline()
        for line in f:
            words = line.split()
            # if len(words) not in [7, 11]:
            #     raise ValueError('数据格式有误')
            col1 = str(words[0])
            l, m = [int(words[1]), int(words[2])]
            if m > lmax: break
            if l > lmax: continue
            value_cs = [float(words[3]), float(words[4])]
            value_cs_std = [float(words[5]), float(words[6])]

            clm[:, l, m] = value_cs
            clm_std[:, l, m] = value_cs_std
    f.close()
    # 物理常数统一
    ref_GM = 3.986004415e14
    ref_R = 6378136.3
    for l in range(lmax):
        factor = (GM / ref_GM) * (R / ref_R) ** l
        clm[:, l, :] = clm[:, l, :] * factor
    # 潮汐系统统一
    if tide_sys == 'zero_tide':
        clm[0, 2, 0] += 4.173e-9
    # convert format C,S into format |C\S|
    cs = clm[0,:,:]
    for n in range(lmax):
        cs[n,n+1:] = clm[1,n+1:,n+1]
    return date, cs, clm_std, lmax


def read_single_model(folder_path=None, lmax=None):
    '''
    function：read all gsm files in the folder
    examples:shc, shc_std, gtime, infos, lmax = read_all_gsm('/home/user/Downloads/CSR_CSR-Release-06_60x60_unfiltered/')
    '''
    filelist = []
    dates = []
    shc, shc_std = [], []
    if folder_path is None:
        sys.exit('未指定GSM文件夹，程序退出')
    if not os.path.exists(folder_path):
        sys.exit('GSM文件夹不存在，程序退出')
    if not os.listdir(folder_path):
        sys.exit("文件夹无GSM文件，程序退出")
    else:
        for (dirname, dirs, files) in walk(folder_path): pass
        # Sort files by month sequences.
        files = np.sort(files)
        for filename in files:
            if filename.endswith('.gfc'):
                filelist.append(path.join(dirname, filename))

        if len(filelist) == 0:
            sys.exit(f'{dirname}文件夹无可读取的GSM文件')

        for fpath in filelist:
            date, clm, clm_std, lmax = read_single_file(fpath, lmax)
            shc.append(clm)
            shc_std.append(clm_std)
            dates.append(date)
        # 纠正重复的时间点
        for i in range(len(dates) - 1):
            if dates[i + 1] == dates[i]:
                ori_date = datetime.datetime.strptime(dates[i + 1], '%Y-%m')
                month = ori_date.month + 1
                if month > 12:
                    new_date = ori_date.replace(month=month % 12, year=ori_date.year + 1)
                else:
                    new_date = ori_date.replace(month=month)
                dates[i + 1] = new_date.strftime("%Y-%m")

        shc, shc_std, dates = np.array(shc), np.array(shc_std), np.array(dates)
        print(f"此时变重力场为{lmax}阶次")
    return shc, shc_std, dates, lmax


def read_existing_models(dirpath):
    """
    function:read the harmonic coefficients in time-varing gavity fields from different institutions at once
    example:model_packs = read_existing_SHs(dirpath)
    """
    # 检查输入的文件夹路径
    if dirpath is None:
        sys.exit('未输入文件夹路径')
    elif not os.path.exists(dirpath):
        sys.exit('文件夹路径不存在')
    elif not os.listdir(dirpath):
        sys.exit('文件夹中无文件')

    # 读取各个时变重力场所在的子文件夹
    model_packs = []

    for f in os.listdir(dirpath):
        # 提取时变场的机构名称
        institution_name = f.split('_')[0]

        # 读取各个机构的时变重力场模型
        subfolder = os.path.join(dirpath, f)
        if os.path.isdir(subfolder):
            shc, _, dates, lmax = read_single_model(subfolder)
            time_series = {'data': shc, 'date': dates, 'degree': lmax, 'name': institution_name + '_SH'}
            model_packs.append(time_series)
            print(f'{institution_name}重力模型读取完成')

    return model_packs
