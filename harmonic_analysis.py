import numpy as np
import math


def harmonic_analysis(data, time):

    # 构建误差方程
    omega = 2 * math.pi
    epochs, colums = data.shape
    A = np.zeros((epochs, 6))
    A[:, 0] = 1
    A[:, 1] = time
    A[:, 2] = np.cos(omega * time)
    A[:, 3] = -np.sin(omega * time)
    A[:, 4] = np.cos(omega * 2 * time)
    A[:, 5] = -np.sin(omega * 2 * time)

    Amplitude1 = np.zeros((colums,))
    Phase1 = np.zeros((colums,))
    Amplitude2 = np.zeros((colums,))
    Phase2 = np.zeros((colums,))
    Resid = np.zeros((epochs, colums))
    Trend = np.zeros((colums,))
    fit_signal = np.zeros((epochs,colums))
    # 利用最小二乘从TWSA中计算长期趋势和周期项
    NM = np.linalg.inv(A.T @ A) @ A.T

    for i in range(colums):
        time_series = data[:, i]
        x = NM @ time_series
        # Exx = np.linalg.inv(A.T @ A) # 计算中误差

        Ampl1 = np.sqrt(x[2] ** 2 + x[3] ** 2)
        if x[3] > 0:  # sin > 0 [0,pi]
            Pha1 = np.arccos(x[2] / Ampl1)
        elif x[3] < 0:  # sin < 0 [pi,2*pi]
            Pha1 = 2 * np.pi - np.arccos(x[2] / Ampl1)
        elif x[3] == 0 & x[2] == 1:
            Pha1 = 0
        elif x[3] == 0 & x[2] == -1:
            Pha1 = np.pi
        elif x[3] == 0 & x[2] == 0:
            Pha1 = 0
        else:
            pass
        Pha1 = np.rad2deg(Pha1)
        # 存储周年振幅和相位
        Amplitude1[i] = Ampl1
        Phase1[i] = Pha1
        # 计算半周年振幅和相位
        Ampl2 = np.sqrt(x[4] ** 2 + x[5] ** 2)
        if x[5] > 0:  # sin > 0 [0,pi]
            Pha2 = np.arccos(x[4] / Ampl2)
        elif x[5] < 0:  # sin < 0 [pi,2*pi]
            Pha2 = 2 * np.pi - np.arccos(x[4] / Ampl2)
        elif x[5] == 0 & x[4] == 1:
            Pha2 = 0
        elif x[5] == 0 & x[4] == -1:
            Pha2 = np.pi
        elif x[5] == 0 & x[4] == 0:
            Pha2 = 0
        else:
            pass
        Pha2 = np.rad2deg(Pha2)
        # 存储半周年振幅和相位
        Amplitude2[i] = Ampl2
        Phase2[i] = Pha2

        # bias[i, j] = x[0]
        res = time_series - A @ x
        fit_signal[:,i]=A @ x
        Resid[:, i] = res
        Trend[i] = x[1]
        if np.any(np.isnan(res)):
            print()


    return fit_signal,Amplitude1, Phase1, Amplitude2, Phase2, Trend, Resid
