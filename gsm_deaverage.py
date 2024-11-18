import sys
import numpy as np


def remove_baseline(cs=None, time=None, start_yr=2004, end_yr=2010):
    # 扣除平均值
    # 示例 cs_anomy, time = remove_baseline(CS_replace, time, start_yr=2004, end_yr=2010)
    if not all(arg is not None for arg in [cs, time, start_yr, end_yr]):
        sys.exit('扣除平均值时缺少参数,程序退出')

    if start_yr >= end_yr:
        sys.exit('起始时间必须小于终止时间,程序退出')
    else:
        if start_yr<=2004 and end_yr>2010:
            indices = np.array(np.where((time >= 2004) & (time < 2010)))
        else:
            indices = np.array(np.where((time >= start_yr) & (time <= end_yr)))

        if np.size(indices) == 0:
            cs = cs - np.average(cs, axis=0)
        else:
            c_mean = np.mean(cs[indices, 0, :, :], axis=1)
            s_mean = np.mean(cs[indices, 1, :, :], axis=1)
            cs[:, 0, :, :] -= c_mean
            cs[:, 1, :, :] -= s_mean

    return cs
