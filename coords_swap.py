import math
import numpy as np


# def longitude_swap(data):
#     # 变换经度，如-180至180变换为0至360
#     n = len(data[0])
#     for i in range(n // 2 + 1):
#         data[:, [i - 1, i - 1 + n // 2]] = data[:, [i - 1 + n // 2, i - 1]]
#     return data

def longitude_swap(data):
    # 变换经度，如-180至180变换为0至360
    n = len(data[0])
    data[:, :] = np.roll(data, math.floor(n/2-1), axis=1)
    return data


def latitude_swap(data):
    # 变换纬度，如-90至90变换为90至-90
    data = np.flipud(data)
    return data
