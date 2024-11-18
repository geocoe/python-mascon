import xarray as xr
import datetime
from grace_time import doy2date
import numpy as np

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

def read_CSR_mascon(CSR_mascon_path):

    # 读取数据
    CSR_mascon_dataset = xr.open_dataset(CSR_mascon_path)
    CSR_mascon = CSR_mascon_dataset['lwe_thickness'].values
    epochs = CSR_mascon_dataset['time'].values

    CSR_mascon = np.flip(CSR_mascon, axis=1)
    doys = []
    for epoch in epochs:
        date = datetime.date(2002, 1, 1) + datetime.timedelta(epoch - 1)
        day = date.timetuple().tm_yday
        year = date.timetuple().tm_year
        # 判断该年是否为闰年
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        # 根据该年是否为闰年，计算该年总天数
        days_per_year = 366 if is_leap_year else 365
        doy = year + day / days_per_year
        doys.append(doy)
    CSR_date = doy2date(doys)
    CSR_date = correct_repeat_time(CSR_date)
    CSR_date = np.array(CSR_date)
    return CSR_mascon,CSR_date