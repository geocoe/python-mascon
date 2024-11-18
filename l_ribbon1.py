import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.sparse.linalg import LinearOperator, eigs
from decimal import Decimal, getcontext

'''
l-ribbon:利用对角化操作快速确定l曲线拐点，进而确定正则化参数
exam: cur_min, cur_max, index = l_ribbon(A, L, k, rp)
输入：
系数矩阵，观测值，对角化层数k，正则化参数区间
输出：
曲率边界，确定的正则化参数序号
ref: Save.2009; Calvetii.2002;
'''


# 将系数矩阵进行三对角化
def lanczos_tridiag(A, y, k):
    #  对称Lanczos算法，用于将对称矩阵A三对角化
    #  输入:
    #    A - 对称矩阵
    #    y - 初始向量
    #    k - 迭代次数
    #  输出:
    #    T - 三对角矩阵
    #    Q - 正交矩阵，其列向量为Lanczos向量
    n = len(y)
    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k + 1)

    # 初始化
    Q[:, 0] = np.squeeze(y) / np.linalg.norm(y)
    q_prev = np.zeros(n)

    for j in range(k):
        # 计算新向量
        z = A @ Q[:, j] - beta[j] * q_prev
        alpha[j] = Q[:, j].T @ z
        z = z - alpha[j] * Q[:, j]

        # 正交化
        if j < k - 1:
            beta[j + 1] = np.linalg.norm(z)
            Q[:, j + 1] = z / beta[j + 1]
            q_prev = Q[:, j]

    # 构造三对角矩阵
    T = np.diag(alpha.astype(float)) + np.diag(beta[1:k].astype(float), 1) + np.diag(beta[1:k].astype(float), -1)

    return T, Q


# 利用残差范数和解范数(&导数）确定曲率
def cur_ribbon(rho_min, rho_max, eta_min, eta_max, eta_de_min, eta_de_max, rp):
    len_rp = len(rp)
    tau_min = np.zeros(len_rp)
    tau_max = np.zeros(len_rp)
    epsilon_min = np.zeros(len_rp)
    epsilon_max = np.zeros(len_rp)
    cur_min = np.zeros(len_rp)
    cur_max = np.zeros(len_rp)

    for i in range(len_rp):
        tau_min[i] = 2 * rho_min[i] * eta_min[i] / (rp[i] ** 2 * eta_max[i] ** 2 + rho_max[i] ** 2) ** 1.5
        tau_max[i] = 2 * rho_max[i] * eta_max[i] / (rp[i] ** 2 * eta_min[i] ** 2 + rho_min[i] ** 2) ** 1.5
        epsilon_min[i] = (rp[i] * rho_min[i] + rp[i] ** 2 * eta_min[i] + rho_max[i] * eta_max[i] / eta_de_max[i])
        epsilon_max[i] = (rp[i] * rho_max[i] + rp[i] ** 2 * eta_max[i] + rho_min[i] * eta_min[i] / eta_de_min[i])
        cur_min[i] = tau_min[i] * epsilon_min[i] if epsilon_min[i] < 0 else tau_max[i] * epsilon_min[i]
        cur_max[i] = tau_min[i] * epsilon_max[i] if epsilon_max[i] < 0 else tau_max[i] * epsilon_max[i]

    return cur_min, cur_max


# 确定曲率边界
# exam: cur_min, cur_max, index = l_ribbon(A, L, k, rp)
# 输入：系数矩阵，观测值，对角化层数，正则化参数区间
# 输出：曲率左边界，右边界。确定的最佳正则化参数序号
def l_ribbon(A, y, k, rp, t):
    ATy = A.T @ y

    # 三对角化

    T_k, _ = lanczos_tridiag(A @ A.T, y, k)
    T_k2, _ = lanczos_tridiag(A.T @ A, ATy, k)

    # cholesky分解，确定高斯正交系数
    cho1_t = cholesky(T_k)
    cho1 = cho1_t.T
    cho1_1 = cho1[:, :k - 1]
    cho2_t = cholesky(T_k2)
    cho2 = cho2_t.T
    cho2_1 = cho2[:, :k - 1]

    # 初始化变量
    iden = np.eye(k)
    len_rp = len(rp)
    rho_max = np.zeros(len_rp)
    rho_min = np.zeros(len_rp)
    eta_max = np.zeros(len_rp)
    eta_min = np.zeros(len_rp)
    eta_de_max = np.zeros(len_rp)
    eta_de_min = np.zeros(len_rp)
    y_norm2 = np.linalg.norm(y) ** 2
    ATy_norm2 = np.linalg.norm(ATy) ** 2

    # 利用三对角化矩阵快速确定解范数（&导数）、残差范数边界
    for i in range(len_rp):
        rho_min[i] = rp[i] ** 2 * y_norm2 * np.dot(iden[0, :],
                                                   np.linalg.solve(cho1 @ cho1.T + rp[i] * iden, iden[:, 0])) ** 2
        rho_max[i] = rp[i] ** 2 * y_norm2 * np.dot(iden[0, :],
                                                   np.linalg.solve(cho1_1 @ cho1_1.T + rp[i] * iden, iden[:, 0])) ** 2
        eta_min[i] = ATy_norm2 * np.dot(iden[0, :], np.linalg.solve(cho2 @ cho2.T + rp[i] * iden, iden[:, 0])) ** 2
        eta_max[i] = ATy_norm2 * np.dot(iden[0, :], np.linalg.solve(cho2_1 @ cho2_1.T + rp[i] * iden, iden[:, 0])) ** 2
        eta_de_min[i] = -2 * ATy_norm2 * np.dot(iden[0, :],
                                                np.linalg.solve(cho2_1 @ cho2_1.T + rp[i] * iden, iden[:, 0])) ** 3
        eta_de_max[i] = -2 * ATy_norm2 * np.dot(iden[0, :],
                                                np.linalg.solve(cho2 @ cho2.T + rp[i] * iden, iden[:, 0])) ** 3

    # 利用残差范数和解范数(&导数）确定曲率
    cur_min, cur_max = cur_ribbon(rho_min, rho_max, eta_min, eta_max, eta_de_min, eta_de_max, rp)
    if cur_min[5] < 0:
        cur_min = -1 * cur_min
        cur_max = -1 * cur_max
    index1 = np.argmax(cur_min)
    index2 = np.argmax(cur_max)
    alpha1 = rp[index1]
    alpha2 = rp[index2]
    index = int((index1 + index2) / 2)
    print(alpha1)
    print(alpha2)
    imgpath = fr'D:\code\point mass\images\Greenland_weighted\RMS_ribbon\{t}.jpg'
    # 绘制曲率边界
    fig = plt.figure(figsize=(10, 5), dpi=400)
    plt.plot(rp, cur_min, 'r--', alpha=1, linewidth=2, label='left-bound')
    plt.plot(rp, cur_max, 'b-', alpha=1, linewidth=1.5, label='right-bound')
    plt.legend()
    fig.savefig(imgpath)
    plt.close()
    # plt.show()

    return cur_min, cur_max, index


def l_curve(A, y, alpha, t):
    num_obs, num_unknown = np.shape(A)
    kum_all = []
    for a in alpha:
        a1 = np.sqrt(a)
        # 横轴
        e = A.T @ y
        si = np.linalg.inv((A.T @ A + a * np.eye(num_unknown)))
        enta = e.T @ si @ si @ e
        # 竖轴
        si1 = np.linalg.inv((A @ A.T + a * np.eye(num_obs)))
        rou = (a ** 2) * (y.T @ si1 @ si1 @ y)
        # 横轴一阶导
        fai = np.linalg.inv((A.T @ A + a * np.eye(num_unknown)))
        fai1 = -4 * a1 * (fai @ fai @ fai)
        enta1 = y.T @ A @ fai1 @ A.T @ y
        # 曲率
        const = np.sqrt(a ** 2 * enta ** 2 + rou ** 2)
        kmu = 2 * enta * rou * (a * enta1 * rou + 2 * a1 * enta * rou + a ** 2 * enta * enta1) / (enta1 * const ** 3)
        kum_all.append(kmu)
    kum_all = np.array(kum_all)
    index = np.argmax(kum_all)
    a_p = alpha[index]
    imgpath = fr'D:\code\point mass\images\Greenland_weighted\RMS_curve\{t}.jpg'
    # 绘制曲率边界
    fig = plt.figure(figsize=(10, 5), dpi=400)
    plt.plot(alpha, kum_all, 'r-', alpha=1, linewidth=2, label='curvature')
    plt.legend()

    fig.savefig(imgpath)
    # plt.show()
    plt.close()
    print(a_p)
    return a_p

