import os
import pdb
import sys
import numpy as np
import xarray as xr
from gsm_deaverage import remove_baseline
from grace_time import find_same_period, date2doy, doy2date
from read_gsm import read_existing_models
from combination_vce import cmb_vce
from harmonic_summation import calculate_TWSA, cs_separate
from output_data import output_mascon, output_TWSA, output_Rmatrix, out_acceleration
from acceleration import acceleration
from generate_mask import region_select, generate_region_mask
from coefficient_set import coefficient_matrix, coeff_matrix_dg
from DisturbGravity import disturbgravity


def vce_and_regularization(parameter):
    y = parameter['observation']
    N_unweighted = parameter['N_unweighted']
    container_A = parameter['container_A']
    epoch = parameter['epoch']
    time = parameter['date']
    R = parameter['R_matrix']
    print(f'开始第{epoch}个月的融合计算')
    (num_model, num_obs, num_unknown) = np.shape(container_A)
    # 各类观测值方差以及验后单位权中误差
    variance = np.ones((num_model,))
    variance_R = 1e22
    scale_factor = 450
    # pdb.set_trace()
    # 正则化参数
    sita = np.ones((num_model,))
    W = np.ones((num_model + 1,))
    # 迭代计数器
    t = 0
    while 1:
        t = t + 1
        print(f"第{epoch}月，第{t}次迭代")
        # 构建法方程系数和常数项
        W_model = sita[0] / variance
        RP0 = sita[0] * scale_factor / variance_R
        # 对个模型的权进行归一化处理
        # W_model = W_model / np.sum(W_model)
        b = np.sum(W_model[:, np.newaxis] * np.sum(container_A * y[:, :, np.newaxis], axis=1), axis=0)
        N_weighted = W_model[:, np.newaxis, np.newaxis] * N_unweighted
        NR = RP0 * R
        N = np.sum(N_weighted, axis=0) + NR
        print(f'第{epoch}个月，第{t}次迭代的法方程构建完成')
        # pdb.set_trace()
        # 法方程逆阵
        N_INV = np.linalg.inv(N)
        # 求解正则化解
        # x = N_INV @ b.T
        x = np.dot(N_INV, b)
        print(f'第{epoch}个月，第{t}次迭代的正则化解求解完成')
        # pdb.set_trace()
        # 计算验后单位权中误差
        V_all_model = np.sum(container_A * x[np.newaxis, np.newaxis, :], axis=2) - y
        V_square = W_model * np.sum(V_all_model ** 2, axis=1)
        sita[:num_model] = V_square / (num_obs - np.sum(N_weighted * N_INV.T[np.newaxis, :, :], axis=(1, 2)))
        print(f'第{epoch}个月，第{t}次迭代验后单位权中误差计算完成')
        # 求解正则化系数
        sita_R = np.dot(np.dot(x.T, NR), x) / (num_unknown - np.trace(NR @ N_INV))
        print(f'第{epoch}个月，第{t}次迭代的正则化参数计算完成')
        end_condition1 = np.abs(np.max(sita) / np.min(sita) - 1)
        end_condition2 = np.abs(np.max(sita) * scale_factor / sita_R - 1)

        if (end_condition1 < 0.1 and end_condition2 < 0.1) or t > 20:
            print(f'第{epoch}个月迭代次数为:{t}')
            print(f'迭代终止时收敛到:{end_condition1}和{end_condition2}')
            print(f'单位权方差为:{sita}')
            print(f'方差为{variance}')
            print(f'正则化参数单位权方差:{sita_R}')
            print(f'正则化参数:{RP0}')
            W[:num_model] = W_model
            W[-1] = RP0
            break
        else:
            variance = sita * (1.0 / W_model)
            variance_R = sita_R / RP0

            print(f'第{epoch}个月，第{t}次迭代未收敛，继续迭代')
            print(variance)
            print(sita_R)
            print(RP0)
    return {'solution': x, 'P': W, 'time': time}


def vce_and_regularization1(parameter):
    y = parameter['observation']
    N_unweighted = parameter['N_unweighted']
    container_A = parameter['container_A']
    epoch = parameter['epoch']
    time = parameter['date']
    R = parameter['R_matrix']
    print(f'开始第{epoch}个月的融合计算')
    (num_model, num_obs, num_unknown) = np.shape(container_A)
    # 验后单位权中误差
    # pdb.set_trace()
    # 正则化参数
    sita = np.ones((num_model + 1,))
    sita_new = np.ones((num_model + 1,))
    W = np.ones((num_model + 1,))
    sita[-1] = 1e18
    scale_factor = 1
    # 迭代计数器
    t = 0
    while 1:
        t = t + 1
        print(f"第{epoch}月，第{t}次迭代")
        # 构建法方程系数和常数项
        sita_initial = sita[0]
        sita_model = sita[:num_model]
        sita_R = sita[-1]
        W_model = sita_initial  / sita_model
        RP0 = sita_initial* scale_factor / sita_R
        # 对个模型的权进行归一化处理
        W_model = W_model / np.sum(W_model)
        # pdb.set_trace()
        b = np.sum(W_model[:, np.newaxis] * np.sum(container_A * y[:, :, np.newaxis], axis=1), axis=0)
        N_weighted = W_model[:, np.newaxis, np.newaxis] * N_unweighted
        NR = RP0 * R
        N = np.sum(N_weighted, axis=0) + NR
        print(f'第{epoch}个月，第{t}次迭代的法方程构建完成')
        # pdb.set_trace()
        # 法方程逆阵
        N_INV = np.linalg.inv(N)
        # 求解正则化解
        x = np.dot(N_INV, b)
        print(f'第{epoch}个月，第{t}次迭代的正则化解求解完成')
        # pdb.set_trace()
        # 计算验后单位权中误差
        V_all_model = np.sum(container_A * x[np.newaxis, np.newaxis, :], axis=2) - y
        V_square = np.sum(V_all_model ** 2, axis=1)
        sita_model = V_square / (num_obs - np.sum(N_weighted * N_INV.T[np.newaxis, :, :], axis=(1, 2)))
        print(f'第{epoch}个月，第{t}次迭代验后单位权中误差计算完成')
        # 求解正则化系数
        sita_R = np.dot(np.dot(x.T, R), x) / (num_unknown - np.trace(NR * N_INV))
        print(f'第{epoch}个月，第{t}次迭代的正则化参数计算完成')
        sita_new[:num_model] = sita_model
        sita_new[-1] = sita_R
        end_condition1 = np.max(np.abs(sita_new / sita - 1))
        # pdb.set_trace()

        if end_condition1 < 0.1 or t > 10:
            print(f'第{epoch}个月迭代次数为:{t}')
            print(f'迭代终止时收敛到:{end_condition1}')
            print(sita)
            print(RP0)
            W[:num_model] = W_model
            W[-1] = RP0
            break
        else:
            sita[:num_model] = sita_model
            sita[-1] = sita_R
            print(f'第{epoch}个月，第{t}次迭代，终止条件为{end_condition1}未收敛，继续迭代')
            print(sita)
            print(RP0)
    return {'solution': x, 'P': W, 'time': time}

def vce_and_regularization2(parameter):
    y = parameter['observation']
    N_unweighted = parameter['N_unweighted']
    container_A = parameter['container_A']
    epoch = parameter['epoch']
    time = parameter['date']
    R = parameter['R_matrix']
    print(f'开始第{epoch}个月的融合计算')
    (num_model, num_obs, num_unknown) = np.shape(container_A)
    # 正则化参数
    alpha = [8e-18, 1e-17, 3e-17, 5e-17, 6e-17, 8e-17, 9e-17, 1e-16, 3e-16, 5e-16, 8e-16, 1e-15]
    xs,ws=[],[]
    # pdb.set_trace()
    for RP0 in alpha:
        sita = np.ones((num_model,))
        sita_new = np.ones((num_model,))
        W = np.ones((num_model+1,))
        # 迭代计数器
        t = 0
        while 1:
            t = t + 1
            print(f"第{epoch}月，第{t}次迭代")
            # 构建法方程系数和常数项
            sita_initial = sita[0]
            sita_model = sita[:num_model]
            W_model = sita_initial / sita_model
            # 对个模型的权进行归一化处理
            W_model = W_model / np.sum(W_model)
            # pdb.set_trace()
            b = np.sum(W_model[:, np.newaxis] * np.sum(container_A * y[:, :, np.newaxis], axis=1), axis=0)
            N_weighted = W_model[:, np.newaxis, np.newaxis] * N_unweighted
            NR = RP0 * R
            N = np.sum(N_weighted, axis=0) + NR
            print(f'第{epoch}个月，第{t}次迭代的法方程构建完成')
            # pdb.set_trace()
            # 法方程逆阵
            N_INV = np.linalg.inv(N)
            # 求解正则化解
            x = np.dot(N_INV, b)
            print(f'第{epoch}个月，第{t}次迭代的正则化解求解完成')
            # pdb.set_trace()
            # 计算验后单位权中误差
            V_all_model = np.sum(container_A * x[np.newaxis, np.newaxis, :], axis=2) - y
            V_square = np.sum(V_all_model ** 2, axis=1)
            sita_model = V_square / (num_obs - np.sum(N_weighted * N_INV.T[np.newaxis, :, :], axis=(1, 2)))
            print(f'第{epoch}个月，第{t}次迭代验后单位权中误差计算完成')
            sita_new[:num_model] = sita_model
            end_condition1 = np.max(np.abs(sita_new / sita - 1))
            # pdb.set_trace()
            if end_condition1 < 0.1 or t > 20:
                print(f'第{epoch}个月迭代次数为:{t}')
                print(f'迭代终止时收敛到:{end_condition1}')
                print(sita)
                print(RP0)
                W[:num_model] = W_model
                W[-1] = RP0
                xs.append(x)
                ws.append(W)
                break
            else:
                sita[:num_model] = sita_model
                print(f'第{epoch}个月，第{t}次迭代，终止条件为{end_condition1}未收敛，继续迭代')
                print(sita)
                print(RP0)
    xs=np.array(xs)
    ws=np.array(ws)

    return {'solution': xs, 'P': ws, 'time': time}

def mascon_fit_method(gravity_models, lon, lat, model_employed, R_matrix_path, constraint_type):
    # 获取未知数个数
    num_unknown = len(lon)
    # 构建正则化矩阵
    if constraint_type == "identity":
        R = np.eye(num_unknown)
        print('使用单位权作为正则化矩阵')
    elif constraint_type == "RMS":
        Rmatrix_file = os.listdir(R_matrix_path)
        if Rmatrix_file:
            R_matrix_path = os.path.join(R_matrix_path, Rmatrix_file[0])
            R_matrix_dataset = xr.open_dataset(R_matrix_path)
            R_vector = R_matrix_dataset['Rmatrix'].values
            R = np.diag(R_vector)
            print('使用RMS作为正则化矩阵')
        else:
            sys.exit('正则化矩阵还未构造')
    else:
        pass
    # pdb.set_trace()
    # 获取球谐系数最大阶数
    single_model = gravity_models[0]
    lmax = single_model['degree']
    # 构建未加权的法方程系数矩阵
    A = coefficient_matrix(lon, lat, lmax)
    N_one = np.dot(A.T, A)
    if model_employed == 'single':
        print('对单个模型进行mascon拟合法计算')
        single_model = gravity_models[1]
        cs = single_model['data']
        lmax = single_model['degree']
        dates = single_model['date']
        time = date2doy(dates)
        # num_month = len(time)
        num_month = 1
        # 转换球谐系数的存储形式
        cs = cs_separate(cs)
        # 除去平均场2004.000-2009.999
        cs_anomalies = remove_baseline(cs, time, start_yr=2004, end_yr=2010)
        print('平均场扣除完成，获取到球谐系数异常')
        x_all_month, variance_all_month, sita_all_month, new_time = [], [], [], []
        for i in range(num_month):
            cnm = cs_anomalies[i, 0, :, :]
            snm = cs_anomalies[i, 1, :, :]
            y_single = np.concatenate((cnm[np.tril_indices(lmax + 1, k=0)], snm[np.tril_indices(lmax + 1, k=0)]))
            y_single = y_single[:, np.newaxis]
            num_obs = len(y_single)
            # 初始化单位权中误差
            sita = np.ones((2,))
            variance = np.ones((2,))
            variance[1] = 1e23
            # 设置迭代计数器
            t = 0
            while 1:
                t = t + 1
                print(f'第{i + 1}个月，第{t}次迭代开始')
                W = sita[0] / variance
                # 构建法方程
                Nb = N_one * W[0]
                NR = W[1] * R
                N = Nb + NR
                b = W[0] * np.dot(A.T, y_single)
                print('法方程构建完成')
                N_INV = np.linalg.inv(N)
                x = np.dot(N_INV, b)
                print('正则化解计算完成')
                V = np.dot(A, x) - y_single
                sita[0] = W[0] * np.dot(V.T, V) / (num_obs - np.trace(np.dot(Nb, N_INV)))
                sita[1] = np.dot(np.dot(x.T, NR), x) / (num_unknown - np.trace(np.dot(NR, N_INV)))
                end_condition = np.abs((np.max(sita) / np.min(sita) - 1))
                # pdb.set_trace()
                if end_condition < 0.1 or t > 10:
                    x_all_month.append(np.squeeze(x))
                    variance_all_month.append(np.squeeze(variance))
                    sita_all_month.append(np.squeeze(sita))
                    new_time.append(time[i])
                    print(f'迭代完成，迭代次数为{t}')
                    print(f'迭代终止时收敛到{end_condition}')
                    print(f'验后单位权方差为：{sita}')
                    print(f'方差{variance}')
                    break
                else:
                    variance = sita * (1 / W)
                    print(f'第{t}迭代不收敛，进行第{t + 1}次迭代')

        x_all_month = np.array(x_all_month)
        variance_all_month = np.array(variance_all_month)
        sita_all_month = np.array(sita_all_month)
        new_time = np.array(new_time)

    if model_employed == 'all':
        print('对多个模型进行mascon拟合法融合计算')
        container_y = []
        for index, single_model in enumerate(gravity_models):
            # 读取GSM文件shc, shc_std, dates, lmax
            cs = single_model['data']
            lmax = single_model['degree']
            dates = single_model['date']
            time = date2doy(dates)
            # 转换球谐系数的存储形式
            cs = cs_separate(cs)
            # 除去平均场2004.000-2009.999
            # pdb.set_trace()
            cs_anomalies = remove_baseline(cs, time, start_yr=2004, end_yr=2010)
            print('平均场扣除完成，获取到球谐系数异常')
            # 构建每个模型各个月的观测值
            num_month = len(time)
            y_all_month = []
            # num_month = 1
            for i in range(num_month):
                cnm = cs_anomalies[i, 0, :, :]
                snm = cs_anomalies[i, 1, :, :]
                y_single = np.concatenate((cnm[np.tril_indices(lmax + 1, k=0)], snm[np.tril_indices(lmax + 1, k=0)]))
                y_all_month.append(y_single)
            y_all_month = np.array(y_all_month)
            container_y.append(y_all_month)
            print(f'第{index + 1}个模型观测值构建完成')
        container_y = np.array(container_y)
        (num_model, num_month, num_obs) = np.shape(container_y)
        # 构建各模型未加权的法方程系数矩阵
        container_A = np.tile(A, (num_model, 1, 1))
        N_unweighted = np.tile(N_one, (num_model, 1, 1))
        print('各模型未加权的法方程系数矩阵构建完成')
        # 逐月融合多个时变场模型计算质量快
        W_all_month, x_all_month, new_time = [], [], []
        # 多进程参数装填
        for k in range(num_month):
            # 切片获取每月各模型观测值
            y = container_y[:, k, :]
            t = time[k]
            para = {'observation': y, 'N_unweighted': N_unweighted, 'container_A': container_A, 'epoch': k + 1,
                    'date': t, 'R_matrix': R}
            # 进行mascon拟合法融合计算
            result = vce_and_regularization1(para)
            x_all_month.append(result['resolution'])
            W_all_month.append(result['P'])
            new_time.append(result['time'])

        x_all_month = np.array(x_all_month)
        W_all_month = np.array(W_all_month)
        new_time = np.array(new_time)

    return x_all_month, W_all_month, new_time


def mascon_calculation(**kwarg):
    # 读取文件夹中所有机构时变重力场模型的信息
    SH_path = kwarg['SH_path']
    mascon_path = kwarg['mascon_path']
    grid_location_path = kwarg['grid_location_path']
    model_employed = kwarg['model_employed']
    R_matrix_path = kwarg['Rmatrix_path']
    constraint_type = kwarg['constraint_type']
    # 读取所有机构的时变重力场模型
    gravity_models = read_existing_models(SH_path)

    # 筛选出时间区间重叠的数据
    start_time, end_time = 2002, 2018
    gravity_models = find_same_period(gravity_models, start_time=start_time, end_time=end_time)

    # 读取参数位置
    grid_location = xr.open_dataset(grid_location_path)
    lon = grid_location['lon'].values
    lat = grid_location['lat'].values
    # mascon拟合法
    TWSA, W, time_model = mascon_fit_method(gravity_models, lon, lat, model_employed,
                                            R_matrix_path, constraint_type)
    mascon_path = os.path.join(mascon_path, f'{model_employed}_TWSA_{constraint_type}.nc')
    print('mascon拟合法计算完成')
    # pdb.set_trace()
    output_mascon(TWSA, W, time_model, lon, lat, mascon_path)


def regularization_matrix(SH_path, DDK_filter_path, grid_location_path, TWSA_path, RMS_mask_path, Rmatrix_path):
    if os.path.exists(TWSA_path) and os.listdir(TWSA_path):
        print('正则化矩阵开始生成')
        # 读取网格经纬度
        location_dataset = xr.open_dataset(grid_location_path)
        lon = location_dataset['lon'].values
        lat = location_dataset['lat'].values
        # 读取陆地水储量
        TWSA_path = os.path.join(TWSA_path, 'TWSA_DDK5_Gaussian100km.nc')
        TWSA_dataset = xr.open_dataset(TWSA_path)
        TWSA = TWSA_dataset['lwe_thickness'].values
        time = TWSA_dataset['time'].values
        date = doy2date(time)

        # 读取掩膜数据
        RMS_mask_path = os.path.join(RMS_mask_path, 'RMS_mask.nc')
        RMS_mask_dataset = xr.open_dataset(RMS_mask_path)
        RMS_mask = RMS_mask_dataset['RMS_mask'].values

        # 进行谐波分析
        # fit_signal, _, _, _, _, _, _ = harmonic_analysis(TWSA, time)
        RMS_signal = np.sqrt(np.mean(TWSA ** 2, axis=0))
        index1 = np.array(np.where((RMS_signal < 4) & (RMS_mask == 1)))
        RMS_signal[index1] = 4

        # index2 = np.array(np.where((RMS_signal>20)&(RMS_mask==1)))
        # RMS_signal[index2] = 20
        RMS_signal[RMS_mask == 0] = 4
        # 将海洋区域大于2cm的调小到2cm，其余保持不变
        # `index = np.array(np.where((RMS_signal > 2) & (RMS_mask == 0)))
        # RMS_signal[index] = 2`
        R = 1 / (RMS_signal ** 2)
        # R = 1/RMS_signal

        # 存储正则化矩阵
        Rmatrix_path = os.path.join(Rmatrix_path, 'Rmatrix.nc')
        output_Rmatrix(R, lon, lat, Rmatrix_path)
        # # # 可视化
        # directory_path = r"D:\code\point mass\images"
        # image_name = f'signal_RMS.jpg'
        # gmt_show_mascon(RMS_signal, lon, lat, directory_path, image_name, date[0])
    else:
        print('进行后处理过程，生成陆地水储量')
        location_dataset = xr.open_dataset(grid_location_path)
        lon = location_dataset['lon'].values
        lat = location_dataset['lat'].values
        # 读取各个时变重力场模型
        gravity_models = read_existing_models(SH_path)
        # 筛选出时间区间重叠的数据
        start_time, end_time = 2002, 2018
        gravity_models = find_same_period(gravity_models, start_time=start_time, end_time=end_time)
        gravity_date = gravity_models[0]['date']
        lmax = gravity_models[0]['degree']
        gravity_time = date2doy(gravity_date)
        # 提取各个机构的球谐系数
        num_models = len(gravity_models)
        SH_datas, SH_names = [], []
        for i in range(num_models):
            SH_datas.append(gravity_models[i]['data'])
            SH_names.append(gravity_models[i]['name'])
        SH_datas = np.array(SH_datas)
        # 使用方差分量估计在解的水平上进行组合
        cmb_solution, _ = cmb_vce(SH_datas)
        # 进行后处理过程，将球谐系数转换成等效水高
        TWSA = calculate_TWSA(cmb_solution, gravity_time, lmax, lon, lat, DDK_filter_path)
        # 输出等效水高
        TWSA_path = os.path.join(TWSA_path, 'TWSA_DDK5_Gaussian100km.nc')
        output_TWSA(TWSA, gravity_time, TWSA_path)


def vce_reg_single(y, A, N_one, R, i):
    num_obs, num_unknown = np.shape(A)
    # 初始化单位权中误差
    sita = np.ones((2,))
    variance = np.ones((2,))
    variance[1] = 1e18

    # 设置迭代计数器
    t = 0
    while 1:
        t = t + 1
        print(f'第{i + 1}个月，第{t}次迭代开始')
        W = sita[0] / variance
        # 构建法方程
        Nb = N_one * W[0]
        NR = W[1] * R
        N = Nb + NR
        b = W[0] * np.dot(A.T, y)
        print('法方程构建完成')
        N_INV = np.linalg.inv(N)
        x = np.dot(N_INV, b)
        print('正则化解计算完成')
        V = np.dot(A, x) - y
        sita[0] = W[0] * np.dot(V.T, V) / (num_obs - np.trace(np.dot(Nb, N_INV)))
        sita[1] = np.dot(np.dot(x.T, NR), x) / (num_unknown - np.trace(np.dot(NR, N_INV)))
        end_condition = np.abs((np.max(sita) / np.min(sita) - 1))
        # pdb.set_trace()
        if end_condition < 0.1 or t > 0:
            print(f'迭代完成，迭代次数为{t}')
            print(f'迭代终止时收敛到{end_condition}')
            print(f'验后单位权方差为：{sita}')
            print(f'方差{variance}')
            return W, x
        else:
            variance = sita * (1 / W)
            print(f'第{t}迭代不收敛，进行第{t + 1}次迭代')


def acceleration_mascon(**kwarg):
    # 读取文件夹中所有机构时变重力场模型的信息
    SH_path = kwarg['SH_path']
    mascon_path = kwarg['mascon_path']
    grid_location_path = kwarg['grid_location_path']
    model_employed = kwarg['model_employed']
    R_matrix_path = kwarg['Rmatrix_path']
    constraint_type = kwarg['constraint_type']
    acceleration_path = kwarg['acceleration_path']
    mask_path = kwarg['RMS_mask_path']
    tn13_path = kwarg['tn13_path']
    tn14_path = kwarg['tn14_path']
    bln_path = kwarg['bln_path']
    Gmask_extended_path = kwarg['Gmask_extended_path']

    # 获取流域名称
    base_name = os.path.basename(bln_path)
    bln_name = os.path.splitext(base_name)[0]
    # 提取指定流域中的经纬度
    grid_dataset = xr.open_dataset(grid_location_path)
    global_lon = grid_dataset['lon']
    global_lat = grid_dataset['lat']
    num_global_grid = len(global_lat)
    # 进行地表质量反演
    if os.path.exists(acceleration_path) and os.listdir(acceleration_path):
        file_name = os.listdir(acceleration_path)
        for file in file_name:
            if file.endswith('.nc'):
                print('以径向加速度作为虚拟观测值，进行点质量计算')
                file_path = os.path.join(acceleration_path, file)
                acc_dataset = xr.open_dataset(file_path)
                lon_d = acc_dataset['lon'].values
                lat_d = acc_dataset['lat'].values
                time = acc_dataset['time'].values
                name = acc_dataset['model_name'].values
                acc = acc_dataset['acc'].values
                num_model, num_time, num_observation = np.shape(acc)
                num_unknown = len(lon_d)
                # 获取掩膜
                mask_file = os.listdir(mask_path)
                mask_file_path = os.path.join(mask_path, mask_file[0])
                mask_dataset = xr.open_dataset(mask_file_path)
                mask = mask_dataset['mask'].values
                mask = np.squeeze(mask)
                # 获取地面点的经纬度
                lat_g = global_lat[mask == 1]
                lon_g = global_lon[mask == 1]
                # 构建正则化矩阵
                if constraint_type == "identity":
                    R = np.eye(num_unknown)
                    print('使用单位权作为正则化矩阵')
                elif constraint_type == "RMS":
                    Rmatrix_file = os.listdir(R_matrix_path)
                    if Rmatrix_file:
                        R_matrix_path = os.path.join(R_matrix_path, Rmatrix_file[0])
                        R_matrix_dataset = xr.open_dataset(R_matrix_path)
                        R_vector = R_matrix_dataset['Rmatrix'].values
                        R_mask = R_vector[mask == 1]
                        R = np.diag(R_mask)
                        print('使用RMS作为正则化矩阵')
                    else:
                        sys.exit('正则化矩阵还未构造')
                # 将角度转换为弧度
                lon_rad_d = np.deg2rad(lon_d)
                lat_rad_d = np.deg2rad(lat_d)
                lon_rad_g = np.deg2rad(lon_g)
                lat_rad_g = np.deg2rad(lat_g)
                A = coeff_matrix_dg(lon_rad_d, lat_rad_d, lon_rad_g, lat_rad_g, num_global_grid)
                N_one = A.T @ A

                # 用单个模型进行点质量计算
                if model_employed == 'single':
                    y_all_month = acc[1, :, :]
                    W_all_month, x_all_month, new_time = [], [], []
                    for i in range(1):
                        y = y_all_month[i, :]
                        t = time[i]
                        W, TWSA = vce_reg_single(y, A, N_one, R, i)
                        W_all_month.append(W)
                        x_all_month.append(TWSA)
                        new_time.append(t)

                # 使用多个模型进行点质量计算
                elif model_employed == 'all':
                    # 构造系数矩阵
                    container_A = np.tile(A, (num_model, 1, 1))
                    N_unweighted = np.tile(N_one, (num_model, 1, 1))
                    print('系数矩阵构造完成')

                    # pdb.set_trace()
                    # 逐月融合多个时变场模型计算质量快
                    W_all_month, x_all_month, new_time = [], [], []
                    for k in range(num_time):
                        # 切片获取每月各模型观测值
                        y = acc[:, k, :]
                        # container_A = container_A[1:,:,:]
                        # N_unweighted = N_unweighted[1:,:,:]
                        t = time[k]
                        para = {'observation': y, 'N_unweighted': N_unweighted, 'container_A': container_A,
                                'epoch': k + 1, 'date': t, 'R_matrix': R}
                        # 进行mascon拟合法融合计算
                        result = vce_and_regularization1(para)
                        x_all_month.append(result['solution'])
                        W_all_month.append(result['P'])
                        new_time.append(result['time'])
                    print('mascon拟合法计算完成')
                    print(name)

                # 输出地表质量变化
                x_all_month = np.array(x_all_month)
                W_all_month = np.array(W_all_month)
                new_time = np.array(new_time)
                mascon_path = os.path.join(mascon_path, f'{model_employed}_TWSA_{constraint_type}_{bln_name}.nc')
                output_mascon(x_all_month, W_all_month, new_time, lon_g, lat_g, mascon_path)
    else:
        # 读取所有机构的时变重力场模型
        gravity_models = read_existing_models(SH_path)

        # 筛选出时间区间重叠的数据
        start_time, end_time = 2002, 2018
        gravity_models = find_same_period(gravity_models, start_time=start_time, end_time=end_time)

        print('从全球网格中筛选出指定流域的经纬度网格')
        # 获取掩膜
        mask_file = os.listdir(Gmask_extended_path)
        mask_file_path = os.path.join(Gmask_extended_path, mask_file[0])
        mask_dataset = xr.open_dataset(mask_file_path)
        mask = mask_dataset['mask'].values
        mask = np.squeeze(mask)
        region_lons = global_lon[mask == 1]
        region_lats = global_lat[mask == 1]

        # 计算各个时变重力场模型的重力扰动位
        print('开始计算重力扰动位')
        observation, time, name = disturbgravity(gravity_models, region_lons, region_lats)

        # 保存加速度为nc文件
        out_acceleration(observation, time, region_lons, region_lats, acceleration_path, bln_name, name)
