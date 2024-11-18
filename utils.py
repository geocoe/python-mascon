import json
import os
import sys
import time
import datetime
import re
from calendar import monthrange
from datetime import datetime
import xarray as xr
def check_ncfile(path):
    if os.path.exists(path):
        files = os.listdir(path)
        file_list=[file for file in files if file.endswith(".nc")]
        file_path = os.path.join(path,file_list[0])
        dataset = xr.open_dataset(file_path)
        return dataset
    else:
        sys.exit('未生成mascon数据')

def select_from_tempValue(obj):
    temp_value_dict = obj["tempValue"][0]
    selected_value = obj["value"]
    selected_key = []
    if int(selected_value) > len(temp_value_dict):
        pass
    else:
        for key, value in temp_value_dict.items():
            if selected_value == value:
                selected_key = key
                break
    if not selected_key:
        # sys.exit(f'参数{obj["name"]}的值无效')
        raise ValueError(f'参数{obj["name"]}的值无效')
    return selected_key


def path_check(path):
    if path is None:
        abs_path = None
    else:
        if os.path.isabs(path):
            # 如果是绝对路径，直接使用原路径
            abs_path = path
        else:
            # 如果是相对路径，转换为绝对路径
            current_directory = os.path.dirname(__file__)
            parent_directory = os.path.dirname(current_directory)
            abs_path = os.path.join(parent_directory, path)
        # 判断操作系统，拼接对应的路径
        if os.name == "nt":
            # Windows系统，使用nt作为路径分隔符
            abs_path = abs_path.replace("/", "\\")
        else:
            # Linux系统，使用posix作为路径分隔符
            abs_path = abs_path.replace("\\", "/")
        # if not os.path.exists(abs_path):
        #     print(f'{path}文件不存在,创建文件夹')
        #     os.makedirs(abs_path)

    return abs_path




def update_return_code(file_path, code):
    # 更新json文件中的return_code
    # 将程序退出状态写入 job_order.json 文件中
    with open(file_path, 'r',encoding='utf-8') as f:
        json_data = json.load(f)

    return_code_found = False
    for status in json_data['Userproperty']['OutputParameter']['ProgramStatus']:
        if status['name'] == 'ReturnCode':
            status['value'] = code
            return_code_found = True
            break

    if not return_code_found:
        # 如果不存在 name 为 "ReturnCode" 的项，则创建新项并设置 value 为 code
        new_status = {
            "name": "ReturnCode",
            "title": "运行状态",
            "type": "int",
            "value": code
        }
        json_data.setdefault('Userproperty', {}).setdefault('OutputParameter', {}) \
            .setdefault('ProgramStatus', []).append(new_status)

    # 将数据写入 job_order.json 文件中
    with open(file_path, 'w', encoding='utf-8') as f:
        f.seek(0)
        json.dump(json_data, f, indent=4, separators=(',', ': '), ensure_ascii=False)
        f.truncate()


def get_time_elapsed(start_time):
    # 返回程序已经运行时间
    return int(time.time() - start_time)


def update_progress_info(file_path, progress):
    # 更新json文件中的进度信息
    with open(file_path, 'r',encoding='utf-8') as f:
        json_data = json.load(f)

    return_code_found = False
    for status in json_data['Userproperty']['OutputParameter']['ProgramStatus']:
        if status['name'] == 'ProgressInfo':
            status['value'] = progress
            return_code_found = True
            break

    if not return_code_found:
        # 如果不存在 name 为 "ReturnCode" 的项，则创建新项并设置 value 为 code
        new_status = {
            "name": "ProgressInfo",
            "title": "进度信息",
            "type": "int",
            "value": progress
        }
        json_data.setdefault('Userproperty', {}).setdefault('OutputParameter', {}) \
            .setdefault('ProgramStatus', []).append(new_status)

    # 将数据写入 job_order.json 文件中
    with open(file_path, 'w', encoding='utf-8') as f:
        f.seek(0)
        json.dump(json_data, f, indent=4, separators=(',', ': '), ensure_ascii=False)
        f.truncate()


def is_chinese(string):
    # 判断错误信息是否是中文
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def update_return_analyse_CHN(file_path, info):
    # 更新json文件中的中文错误信息
    with open(file_path, 'r',encoding='utf-8') as f:
        json_data = json.load(f)

    return_code_found = False
    for status in json_data['Userproperty']['OutputParameter']['ProgramStatus']:
        if status['name'] == 'ReturnAnalyseCHN':
            status['value'] = info
            return_code_found = True
            break

    if not return_code_found:
        # 如果不存在 name 为 "ReturnCode" 的项，则创建新项并设置 value 为 code
        new_status = {
            "name": "ReturnAnalyseCHN",
            "title": "中文错误描述",
            "type": "string",
            "value": info
        }
        json_data.setdefault('Userproperty', {}).setdefault('OutputParameter', {}) \
            .setdefault('ProgramStatus', []).append(new_status)

    # 将数据写入 job_order.json 文件中
    with open(file_path, 'w', encoding='utf-8') as f:
        f.seek(0)
        json.dump(json_data, f, indent=4, separators=(',', ': '), ensure_ascii=False)
        f.truncate()


def update_return_analyse_ENG(file_path, info):
    # 更新json文件中的英文错误信息
    with open(file_path, 'r',encoding='utf-8') as f:
        json_data = json.load(f)

    return_code_found = False
    for status in json_data['Userproperty']['OutputParameter']['ProgramStatus']:
        if status['name'] == 'ReturnAnalyseENG':
            status['value'] = info
            return_code_found = True
            break

    if not return_code_found:
        # 如果不存在 name 为 "ReturnCode" 的项，则创建新项并设置 value 为 code
        new_status = {
            "name": "ReturnAnalyseENG",
            "title": "英文错误描述",
            "type": "string",
            "value": info
        }
        json_data.setdefault('Userproperty', {}).setdefault('OutputParameter', {}) \
            .setdefault('ProgramStatus', []).append(new_status)

    # 将数据写入 job_order.json 文件中
    with open(file_path, 'w', encoding='utf-8') as f:
        f.seek(0)
        json.dump(json_data, f, indent=4, separators=(',', ': '), ensure_ascii=False)
        f.truncate()


def ENG_to_CHN(info):
    # 将英文错误信息转为中文
    if "ValueError" in info:
        msg = '值错误'
    elif "SyntaxError" in info:
        msg = '系统错误'
    elif "NameError" in info:
        msg = '函数或变量名错误'
    elif "TypeError" in info:
        msg = '类型错误'
    elif "IndexError" in info:
        msg = '索引错误'
    elif "AttributeError" in info:
        msg = '访问的对象缺少属性'
    elif "MemoryError" in info:
        msg = '内存错误'
    elif "KeyError" in info:
        msg = '键错误'
    else:
        msg = '其他错误'

    return msg


def check_path_exist(path):
    # 检验路径是否存在
    if not os.path.exists(path):
        return False
    else:
        return True


def is_valid_date(date_string):
    # 判断输入的字符串是否满足“2002-04-01”的形式，并且天数是否小于当月的最大天数
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(pattern, date_string):
        return False

    year, month, day = map(int, date_string.split('-'))

    if month > 12:
        return False

    _, max_day = monthrange(year, month)
    if day > max_day:
        return False

    return True


def get_fractional_year(date_string):
    try:
        # 将日期字符串转换为 datetime 对象
        date_obj = datetime.strptime(date_string, '%Y-%m-%d')

        # 计算该日期在该年中是第几天
        day_of_year = date_obj.timetuple().tm_yday

        # 计算该年的总天数
        is_leap_year = (date_obj.year % 4 == 0 and date_obj.year % 100 != 0) or date_obj.year % 400 == 0
        total_days = 366 if is_leap_year else 365

        # 计算年+day_of_year / 总天数的值
        year_decimal = date_obj.year + day_of_year / total_days

        return year_decimal

    except ValueError:
        return None
