import datetime
import numpy as np
import os
import sys
from grace_time import doy2date


def is_leap_year(year):
    dpm_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dpm_stnd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 4000 != 0):
        return np.array(dpm_leap, dtype=np.float64)
    else:
        return np.array(dpm_stnd, dtype=np.float64)


def read_tn13(filename=None):
    '''
    function: Read C10, C11, S11
    input: Path of TN13 file
    example:C10, C11, S11, time = read_TN13('/home/user/Downloads/TN-13_GEOC_CSR_RL06.txt')
    '''
    c1 = []
    s1 = []
    time = []

    if filename is None:
        sys.exit('未指定TN13文件路径,程序退出')
    if not os.path.exists(filename):
        sys.exit('TN13文件不存在，程序退出')
    if os.stat(filename).st_size == 0:
        sys.exit('TN13文件为空，程序退出')

    expected_first_line = 'GRACE Technical Note 13'
    with open(filename, "r") as file:
        first_line = file.readline().rstrip()
    if first_line != expected_first_line:
        sys.exit('TN13文件格式不匹配，程序退出')
    file.close()

    with open(filename, 'r') as fid:
        keywords = 'end of header'
        head_index = 0
        for line in fid:
            head_index += 1
            if keywords in line:
                break
    fid.close()

    with open(filename, "r") as f:
        for i in range(head_index):
            f.readline()
        for line in f:
            words = line.split()
            clm1 = float(words[3])
            slm1 = float(words[4])

            start_date = datetime.datetime.strptime(str(words[7].split('.')[0]), '%Y%m%d').date()
            end_date = datetime.datetime.strptime(str(words[8].split('.')[0]), '%Y%m%d').date()

            start_doy = start_date.timetuple().tm_yday
            end_doy = end_date.timetuple().tm_yday

            dpy_start = is_leap_year(int(start_date.year)).sum()
            dpy_end = is_leap_year(int(end_date.year)).sum()
            end_cyclic = ((int(end_date.year) - int(start_date.year)) * dpy_start + end_doy)
            mid_day = np.mean([start_doy, end_cyclic])

            if mid_day <= dpy_start:
                d1time = int(start_date.year) + mid_day / dpy_start
            else:
                d1time = int(end_date.year) + (mid_day - dpy_start) / dpy_end
            c1.append(clm1)
            s1.append(slm1)
            time.append(d1time)
    f.close()
    c10 = c1[::2]
    c11 = c1[1::2]
    s11 = s1[1::2]
    time = time[1::2]
    return c10, c11, s11, time

def correct_repeat_time(date):
    # 纠正重复的时间点
    for i in range(len(date) - 1):
        if date[i + 1] == date[i]:
            ori_date = datetime.datetime.strptime(date[i + 1], '%Y-%m')
            month = ori_date.month + 1
            if month > 12:
                new_date = ori_date.replace(month=month % 12, year=ori_date.year + 1)
            else:
                new_date = ori_date.replace(month=month)
            date[i + 1] = new_date.strftime("%Y-%m")
    return date
def read_tn14(filename=None):
    '''
    function: Read C20, C30
    input: Path of TN14 file
    example: C20, C30, time = read_TN14('/home/user/Downloads/TN-14_C30_C20_GSFC_SLR.txt')
    '''
    c20 = []
    c30 = []
    time = []

    if filename is None:
        sys.exit('未指定TN14文件路径,程序退出')
    if not os.path.exists(filename):
        sys.exit('TN14文件路径不存在，程序退出')
    if os.stat(filename).st_size == 0:
        sys.exit('TN14文件为空，程序退出')

    expected_first_line = 'Title: NASA GSFC SLR C20 and C30 solutions'
    with open(filename, "r") as file:
        first_line = file.readline().rstrip()
    if first_line != expected_first_line:
        sys.exit('TN14文件格式不匹配，程序退出')
    file.close()

    with open(filename, 'r') as fid:
        keywords = 'Product:'
        head_index = 0
        for line in fid:
            head_index += 1
            if keywords in line:
                break
    fid.close()

    with open(filename, "r") as f:
        for i in range(head_index):
            f.readline()
        for line in f:
            words = line.split()
            clm2 = float(words[2])
            clm3 = float(words[5])
            start_time = float(words[1])
            end_time = float(words[9])
            mid_time = np.mean([start_time, end_time])
            c20.append(clm2)
            c30.append(clm3)
            time.append(mid_time)
    f.close()
    return c20, c30, time


def replace_shc(shc=None, time=None, tn13=None, tn14=None):
    # example: CS_replace, time = replace_shc(shc, time,
    # '/home/user/Downloads/TN-13_GEOC_CSR_RL06.txt', '/home/user/Downloads/TN-14_C30_C20_GSFC_SLR.txt')
    cs_replace = shc
    if not all(arg is not None for arg in [shc, time, tn13, tn14]):
        sys.exit('替换低阶项参数不完备,程序退出')
    else:
        C10, C11, S11, time_geoc = read_tn13(tn13)
        C20, C30, time_slr = read_tn14(tn14)

        # replacing degree 1
        for ii in range(time.shape[0]):
            time_gsm = time[int(ii)]
            for jj in range(len(time_geoc)):
                time_1 = time_geoc[int(jj)]
                if abs(time_gsm - time_1) <= 0.02:
                    cs_replace[int(ii), 0, 1, 0] = C10[jj]
                    cs_replace[int(ii), 0, 1, 1] = C11[jj]
                    cs_replace[int(ii), 1, 1, 1] = S11[jj]

        # replacing C20 and C30
        for uu in range(time.shape[0]):
            time_gsm = time[int(uu)]
            for vv in range(len(time_slr)):
                time_2 = time_slr[int(vv)]
                if abs(time_gsm - time_2) <= 0.02:
                    cs_replace[int(uu), 0, 2, 0] = C20[vv]
                    if np.isnan(C30[vv]):
                        continue
                    else:
                        cs_replace[int(uu), 0, 3, 0] = C30[vv]

    return cs_replace
def replace_shc_rewrite(shc=None, dates=None, tn13=None, tn14=None):
    cs_replace = shc
    if not all(arg is not None for arg in [shc, dates, tn13, tn14]):
        sys.exit('替换低阶项参数不完备,程序退出')
    C10, C11, S11, term1_time = read_tn13(tn13)
    C20, C30, term2_time = read_tn14(tn14)
    term1_date = doy2date(term1_time)
    term2_date = doy2date(term2_time)
    term1_date = correct_repeat_time(term1_date)
    term2_date = correct_repeat_time(term2_date)
    # 低阶项替换
    for index, date in enumerate(dates):
        if date in term1_date:
            search_index = term1_date.index(date)
            cs_replace[index, 0, 1, 0] = C10[search_index]
            cs_replace[index, 0, 1, 1] = C11[search_index]
            cs_replace[index, 1, 1, 1] = S11[search_index]
        else:
            pass

        if date in term2_date:
            search_index = term2_date.index(date)
            cs_replace[index, 0, 2, 0] = C20[search_index]
            if np.isnan(C30[search_index]):
                pass
            else:
                cs_replace[index, 0, 3, 0] = C30[search_index]
        else:
            pass
    return cs_replace