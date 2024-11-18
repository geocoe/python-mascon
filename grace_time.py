import math
import datetime
import sys
import numpy as np
import pandas as pd


def is_leap_year(year):
    # 判断是否是闰年，并返回每个月的天数
    dpm_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dpm_stnd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 4000 != 0):
        return np.array(dpm_leap, dtype=np.float64)
    else:
        return np.array(dpm_stnd, dtype=np.float64)


def doy2month(time):
    # 将GRACE时间转为年-月
    years = []
    months = []
    for i in range(len(time)):
        dates = time[i]
        # 年份
        year = math.floor(dates)
        # 判断该年是否为闰年
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        # 根据该年是否为闰年，计算该年总天数
        days_per_year = 366 if is_leap_year else 365
        # 根据输入的时间计算day of year
        day_of_year = (dates - year) * days_per_year

        # 计算对应的日期
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)

        # 获取日期对应的月份
        month = date.month

        years.append(year)
        months.append(month)

    years = np.array(years)
    months = np.array(months)

    # 返回日期
    return years, months


def month2doy(years, months):
    # 将年-月转换为day of year
    # 取每个月的mid day完成转换
    dates = []
    for year, month in zip(years, months):
        # 计算中间一天的日期
        mid_day = datetime.date(year, month, 15)
        # 获取该日期对应的day of year
        day_of_year = mid_day.timetuple().tm_yday

        # 判断该年是否为闰年
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        # 根据该年是否为闰年，计算该年总天数
        days_per_year = 366 if is_leap_year else 365

        date = year + day_of_year / days_per_year
        dates.append(date)

    dates = np.array(dates)
    return dates


def find_gaps(years, months):
    start_year = years[0]
    start_month = months[0]
    end_year = years[-1]
    end_month = months[-1]

    # 转换为pandas中的日期类型，起始时间为每个月的第一天
    start_date = pd.to_datetime(f"{start_year}-{start_month:02}-01")

    # 转换为pandas中的日期类型，结束时间为每个月的最后一天
    end_date = pd.to_datetime(f"{end_year}-{end_month:02}-01") + pd.offsets.MonthEnd(1) - pd.offsets.Day(1)

    # 生成时间序列，并将日期格式化为'%Y-%m'的字符串列表
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    date_input = [f"{year}-{month:02d}" for year, month in zip(years, months)]

    set1 = set(date_input)
    set2 = set(date_range)

    # # 分别获取唯一和重复的月份
    # unique_lst = list(set(date_input))
    # counter = Counter(date_input)
    # duplicated_lst = [k for k, v in counter.items() if v > 1]

    # 判断差异
    diff = set2.difference(set1)
    if len(diff) > 0:
        # 获取差异在原先的列表中的索引
        missing_idx = np.array([date_range.index(elem) for elem in diff])
        missing_idx = np.sort(missing_idx)
        # print(f"{[date_range[i] for i in missing_idx.tolist()]} 的数据缺失")
        print(f"GRACE存在{np.array(date_range)[missing_idx]} 的空缺")

    else:
        print("GRACE无空缺")
        missing_idx = []

    return missing_idx, np.array(date_range)[missing_idx]


def check_time_span(year_obs, month_obs, year_model, month_model, shift=0):
    # 检查GLDAS数据是否完全覆盖GRACE的观测时间段
    # 参数shift表示GLDAS相对于GRACE提前的月数，如提前3个月，shift=3，缺省值为0
    if shift < 0:
        sys.exit('参数shift必须大于0')
    else:
        pass

    # 获取GRACE的起止年月
    grace_start_year = year_obs[0]
    grace_start_month = month_obs[0]
    grace_end_year = year_obs[-1]
    grace_end_month = month_obs[-1]

    # 将GRACE的开始时间提前若干个月
    grace_start_month = grace_start_month - shift
    if grace_start_month <= 0:
        grace_start_month += 12
        grace_start_year -= 1

    # 转换为pandas中的日期类型，起始时间为每个月的第一天
    start_date = pd.to_datetime(f"{grace_start_year}-{grace_start_month:02}-01")

    # 转换为pandas中的日期类型，结束时间为每个月的最后一天
    end_date = pd.to_datetime(f"{grace_end_year}-{grace_end_month:02}-01") + pd.offsets.MonthEnd(1) - pd.offsets.Day(1)

    # 生成时间序列，并将日期格式化为'%Y-%m'的字符串列表
    grace_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    gldas_range = [f"{year}-{month:02d}" for year, month in zip(year_model, month_model)]

    set1 = set(gldas_range)
    set2 = set(grace_range)
    # 判断差异
    diff = set2.difference(set1)
    if len(diff) > 0:
        # 获取差异在原先的列表中的索引
        missing_idx = np.array([grace_range.index(elem) for elem in diff])
        missing_idx = np.sort(missing_idx)
        print(f"GLDAS存在 {np.array(grace_range)[missing_idx]} 的数据缺失")
        return False
    else:
        print("GLDAS无数据缺失")
        return True


def date2doy(date_list):
    years, months = [], []
    for date in date_list:
        date_obj = datetime.datetime.strptime(date, '%Y-%m')
        year = date_obj.year
        month = date_obj.month
        years.append(year)
        months.append(month)
    days = month2doy(years, months)
    return days


def doy2date(days):
    dates = []
    years, months = doy2month(days)
    dates = [f'{year}-{month:02}' for year, month in zip(years, months)]
    return dates


def find_same_period(gra_data, mas_data=None, start_time=2002, end_time=2017):
    # 判断球谐系数和mascon是否同时被筛选
    if mas_data == None:
        cmb_data = gra_data
    else:
        cmb_data = gra_data + mas_data

    # 时变重力场的数量和质量块的数量
    num = len(cmb_data)
    num_gra = len(gra_data)
    date_ranges = []
    for i in range(num):
        if 'date' in cmb_data[i]:
            date_range = cmb_data[i]['date']
        elif 'time' in cmb_data[i]:
            date_range = doy2date(cmb_data[i]['time'])
        else:
            sys.exit('没有时间帧')
        date_ranges.append(date_range)
    # 筛选出时间重叠的数据
    base_set = set(date_ranges[0])
    for j in range(num):
        temp_set = base_set & set(date_ranges[j])
        base_set = temp_set

    # 截取指定时间段的数据
    base_set = list(base_set)
    base_set = date2doy(base_set)
    base_set = base_set[(base_set > start_time) & (base_set < end_time)]
    base_set = doy2date(base_set)
    print(f'数据交集个数{len(base_set)}')
    set1 = set(date_ranges[1])
    diff_set = set1.difference(base_set)
    for m in range(num):
        # index = [date_ranges[m].index(elem) for elem in enumerate(base_set)]
        indeces = [index for index, elem in enumerate(date_ranges[m]) if elem in base_set]
        indeces = np.sort(indeces)
        data = cmb_data[m]['data'][indeces]
        if 'date' in cmb_data[m]:
            date = cmb_data[m]['date'][indeces]
        elif 'time' in cmb_data[m]:
            date = cmb_data[m]['time'][indeces]
        else:
            pass

        if 'gra' in cmb_data[m]['name'] or 'SH' in cmb_data[m]['name']:
            for n in range(num_gra):
                if gra_data[n]['name'] == cmb_data[m]['name']:
                    gra_data[n]['data'] = data
                    if 'date' in gra_data[n]:
                        gra_data[n]['date'] = date
                    else:
                        gra_data[n]['time'] = date


        elif 'MC' in cmb_data[m]['name']:
            for k in range(num - num_gra):
                if mas_data[k]['name'] == cmb_data[m]['name']:
                    mas_data[k]['data'] = data
                    mas_data[k]['date'] = date
    if mas_data == None:
        return gra_data
    else:
        return gra_data, mas_data


def same_period_from_both(base_time, model_data, model_time):
    if isinstance(base_time[0], str):
        temp_base_time = base_time
    else:
        temp_base_time = doy2date(base_time)
    if isinstance(model_time[0], str):
        temp_model_time = model_time
    else:
        temp_model_time = doy2date(model_time)
    # 判断数据中是否包含了所有的grace时间点
    diff_set = set(temp_base_time).difference(set(temp_model_time))
    if len(diff_set) != 0:
        sys.exit('该数据中未包含所有的grace时间点')
    else:
        index = [i for i, item in enumerate(temp_model_time) if item in temp_base_time]
        new_model_time = model_time[index]
        new_model_data = model_data[index]
    return new_model_data, new_model_time


def time_elapsed_show(start_time,end_time):
    # 程序消耗时间
    time_consumed = end_time - start_time
    # 转换为时分秒格式
    seconds = int(time_consumed)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    print(f'程序运行时间为{hours:02d}:{minutes:02d}:{seconds:02d}')
