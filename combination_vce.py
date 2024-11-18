import numpy as np
def cmb_vce(grid_data):
    # 获得网格数据各个维度的长度
    num_solu, num_time, rows, colums = grid_data.shape
    print('开始进行vce组合')
    # 给最终的组合解和权重结果分配空间
    cmb_solus, wgt_solus = [], []
    # 开始进行时变重力场的组合
    for i in range(num_time):
        data = grid_data[:, i, :, :]
        # 定义时变重力场的初始权重
        wgt_solu = np.ones((num_solu,)) * (1 / num_solu)
        # 设置迭代器
        t = 0
        # 方差分量估计定权，进行迭代
        while 1:
            t = t + 1
            # 获取组合解
            cmb_solu = np.sum(wgt_solu[:, np.newaxis, np.newaxis] * data, axis=0)
            # 根据方差分量估计定权
            size_solu = np.size(cmb_solu[~np.isnan(cmb_solu)])
            rsd_solu = cmb_solu[np.newaxis, ...] - data
            r_solu = size_solu * (1 - wgt_solu)
            rsd_square = wgt_solu[:, np.newaxis, np.newaxis] * rsd_solu ** 2
            rsd_square = np.where(np.isnan(rsd_square), 0, rsd_square)
            sigma_solus = np.sum(rsd_square, axis=(1, 2)) / np.squeeze(r_solu)

            if t > 50 or np.abs(np.max(sigma_solus) / np.min(sigma_solus) - 1) < 1e-4:
                wgt_solus.append(wgt_solu)
                cmb_solus.append(cmb_solu)
                break
            else:
                wgt_solu = wgt_solu / sigma_solus
                # 对权进行归一化处理
                wgt_solu = wgt_solu / sum(wgt_solu)
    cmb_solus, wgt_solus = np.array(cmb_solus), np.array(wgt_solus)
    print('vce组合完成')
    return cmb_solus, wgt_solus