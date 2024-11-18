import numpy as np

'''
最小MSE确定正则化参数
'''

# 求正则化估计下的无偏验后单位权中误差.ref:Kunpu JI.2022.JGR
def sigma2_tik(L, A, alpha):
    m, n = A.shape
    U, S, _ = np.linalg.svd(A)
    # 由于奇异值分解形式与MATLAB不一致，故需调整
    U = np.mat(U)
    F1 = np.zeros((m, m))
    F2 = 0
    Im = np.eye(m)
    for i in range(n):
        F1 += (S[i] ** 2 * (S[i] ** 2 + 2 * alpha) / (S[i] ** 2 + alpha) ** 2) * np.dot(U[:, i], U[:, i].T)
        F2 += alpha ** 2 / (S[i] ** 2 + alpha) ** 2
    F12 = np.linalg.norm((Im - F1) @ L) ** 2
    sigma2 = F12 / (m - n + F2)
    return sigma2

# 求MSE导数
def func1(A, alpha, x0, sigma2):
    H = 0
    _, n = A.shape
    _, S, V1 = np.linalg.svd(A)
    V = np.mat(V1.T)
    for i in range(n):
        H += S[i] ** 2 * (alpha * (V[:, i].T @ x0) ** 2 - sigma2) / (S[i] ** 2 + alpha) ** 3
    return H

'''
MSE最小确定最小正则化参数，
输入：系数矩阵、单位权中误差、未知参数近似值、迭代收敛条件（两次正则化参数差值）
输出：正则化参数
'''
def minMSE_NIT(A, sigma2, xotik, delta):
    alpha0 = 0
    _, n = A.shape
    flag = 1
    alpha1 = 1e-14
    while flag:
        H = func1(A, alpha1, xotik, sigma2)
        if H < 0:
            alpha1 = 2 * alpha1
        else:
            flag = 0

    flag = 1

    while flag:
        alphaT = (alpha0 + alpha1) / 2
        H = func1(A, alphaT, xotik, sigma2)
        if abs(H) < delta or abs(alpha1 - alpha0) < delta:
            flag = 0
        elif H < 0:
            alpha0 = alphaT
        else:
            alpha1 = alphaT

    minMSE_alpha = alphaT
    return minMSE_alpha

'''
迭代MSE最小确定最小正则化参数，
输入：观测值、系数矩阵、单位权方差、迭代收敛条件（1：两次正则化参数差值；2：迭代次数）
输出：正则化参数
'''
def minMSE_IT(L, A, sigma2, xotik, delta, T):
    In = np.eye(A.shape[1])
    for i in range(T):
        alpha0 = minMSE_NIT(A, sigma2, xotik, delta)
        xotik = np.linalg.solve(A.T @ A + alpha0 * In, A.T @ L)
        sigma2 = sigma2_tik(L, A, alpha0)

    minMSEalpha = alpha0
    return minMSEalpha
