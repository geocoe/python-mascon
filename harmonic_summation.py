from gsm_deaverage import remove_baseline
from gc2mc import shc2mc
from filtering_gsm import filter_gaussian
import numpy as np
from ddk_filter import filter_ddk
from associated_legendre import plm_holmes
from tqdm import tqdm


def cs_separate(cs):
    if cs.ndim == 2:
        rows, colums = np.shape(cs)
        snm = np.zeros((rows, colums))
        cnm = np.tril(cs)
        for i in range(rows - 1):
            snm[i + 1:, i + 1] = cs[i, i + 1:]
        return cnm, snm
    elif cs.ndim == 3:
        epoch, rows, colums = np.shape(cs)
        cs_new = []
        for i in range(epoch):
            shcs = cs[i, :, :]
            cnm0 = np.tril(shcs)
            snm0 = np.zeros((rows, colums))
            for j in range(rows - 1):
                snm0[j + 1:, j + 1] = shcs[j, j + 1:]
            cs0 = [cnm0, snm0]
            cs_new.append(cs0)
        cs_new = np.array(cs_new)
        return cs_new


def harmonic_summation(clm1, slm1, lon, lat, LMIN=0, LMAX=60, MMAX=None, PLM=None):
    """
    Converts data from spherical harmonic coefficients to a spatial field

    Parameters
    ----------
    clm1: float
        cosine spherical harmonic coefficients in output units
    slm1: float
        sine spherical harmonic coefficients in output units
    lon: float
        longitude array
    lat: float
        latitude array
    LMIN: int, default 0
        Lower bound of Spherical Harmonic Degrees
    LMAX: int, default 60
        Upper bound of Spherical Harmonic Degrees
    MMAX: int or NoneType, default None
        Upper bound of Spherical Harmonic Orders
    PLM: float or NoneType, default None
        Fully-normalized associated Legendre polynomials

    Returns
    -------
    spatial: float
        spatial field (lon * lat)
    """

    # if LMAX is not specified, will use the size of the input harmonics
    if LMAX == 0:
        LMAX = np.shape(clm1)[0] - 1
    # upper bound of spherical harmonic orders (default = LMAX)
    if MMAX is None:
        MMAX = np.copy(LMAX)

    # 角度转为弧度
    # Longitude in radians
    phi = (np.squeeze(lon) * np.pi / 180.0)[np.newaxis, :]
    # colatitude in radians
    th = (90.0 - np.squeeze(lat)) * np.pi / 180.0
    thmax = len(th)

    # if plms are not pre-computed: calculate Legendre polynomials
    if PLM is None:
        PLM, dPLM = plm_holmes(LMAX, np.cos(th))

    # Truncating harmonics to degree and order LMAX
    # removing coefficients below LMIN and above MMAX
    mm = np.arange(0, MMAX + 1)
    clm = np.zeros((LMAX + 1, MMAX + 1))
    slm = np.zeros((LMAX + 1, MMAX + 1))
    clm[LMIN:LMAX + 1, mm] = clm1[LMIN:LMAX + 1, mm]
    slm[LMIN:LMAX + 1, mm] = slm1[LMIN:LMAX + 1, mm]
    # Calculate fourier coefficients from legendre coefficients
    d_cos = np.zeros((MMAX + 1, thmax))  # [m,th]
    d_sin = np.zeros((MMAX + 1, thmax))  # [m,th]
    for k in range(0, thmax):
        # summation over all spherical harmonic degrees
        d_cos[:, k] = np.sum(PLM[:, mm, k] * clm[:, mm], axis=0)
        d_sin[:, k] = np.sum(PLM[:, mm, k] * slm[:, mm], axis=0)

    # Final signal recovery from fourier coefficients
    m = np.arange(0, MMAX + 1)[:, np.newaxis]
    # Calculating cos(m*phi) and sin(m*phi)
    ccos = np.cos(np.dot(m, phi))
    ssin = np.sin(np.dot(m, phi))
    # summation of cosine and sine harmonics
    # s = np.dot(np.transpose(ccos), d_cos) + np.dot(np.transpose(ssin), d_sin)
    s = np.sum(ccos * d_cos, axis=0) + np.sum(ssin * d_sin, axis=0)

    # return output data
    return s


def calculate_TWSA(SH_solution, SH_time, lmax, lon, lat, DDK_filter_path):
    # 转换球谐系数的存储形式|c\s|->c,s
    cs = cs_separate(SH_solution)
    # 去2004.000~2009.999平均场
    cs_anomy = remove_baseline(cs, SH_time, 2004, 2010)
    # geoid系数转换为等效水高系数
    mc = shc2mc(cs_anomy, 'Water', lmax)

    # DDK5
    wd = "DDK5"
    mc = filter_ddk(wd, mc, DDK_filter_path)
    # mc = destriping_gsm(mc, locals().get('destripe_method'))
    print(f"已完成{wd}滤波")

    # 高斯滤波
    radius_filter = 100
    mc = filter_gaussian(mc, radius_filter)
    print(f'{radius_filter}km的高斯滤波完成')
    # 球谐综合
    clm = mc[:, 0, :, :]
    slm = mc[:, 1, :, :]
    lwe = []
    for c, s in tqdm(zip(clm, slm), total=len(SH_time), desc='等效水高计算中'):
        tmp = harmonic_summation(c, s, lon, lat, LMAX=lmax)
        lwe.append(tmp)
    lwe = np.array(lwe) * 100

    return lwe


def spherical_analysis(data, lon, lat, LMAX=60, MMAX=None, PLM=0, resolution=1):
    """
    Converts data from the spatial domain to spherical harmonic coefficients

    Parameters
    ----------
    data: float
        data magnitude
    lon: float
        longitude array
    lat: float
        latitude array
    LMAX: int, default 60
        Upper bound of Spherical Harmonic Degrees
    MMAX: int or NoneType, default None
        Upper bound of Spherical Harmonic Orders
    PLM: float, default 0
        input Legendre polynomials

    Returns
    -------
    clm: float
        cosine spherical harmonic coefficients
    slm: float
        sine spherical harmonic coefficients
    l: int
        spherical harmonic degree to LMAX
    m: int
        spherical harmonic order to MMAX
    """

    # dimensions of the longitude and latitude arrays
    nlon = np.int64(len(lon))
    nlat = np.int64(len(lat))
    # grid step
    dlon = np.abs(resolution)
    dlat = np.abs(resolution)
    # longitude degree spacing in radians
    dphi = dlon * np.pi / 180.0
    # colatitude degree spacing in radians
    dth = dlat * np.pi / 180.0

    # reformatting longitudes to range 0:360 (if previously -180:180)
    if np.count_nonzero(lon < 0):
        lon[lon < 0] += 360.0
    # calculate longitude and colatitude arrays in radians
    phi = np.reshape(lon, (1, nlon)) * np.pi / 180.0  # reshape to 1xnlon
    th = (90.0 - np.squeeze(lat)) * np.pi / 180.0  # remove singleton dimensions

    # Calculating cos/sin of phi arrays (output [m,phi])
    # LMAX+1 as there are LMAX+1 elements between 0 and LMAX
    m = np.arange(MMAX + 1)[:, np.newaxis]
    ccos = np.cos(np.dot(m, phi))
    ssin = np.sin(np.dot(m, phi))

    # Multiplying sin(th) with differentials of theta and phi
    # to calculate the integration factor at each latitude
    int_fact = np.sin(th) * dphi * dth
    coeff = 1.0 / (4.0 * np.pi)

    # Calculate polynomials using Holmes and Featherstone (2002) relation
    plm = np.zeros((LMAX + 1, MMAX + 1, nlat))
    if (np.ndim(PLM) == 0):
        plmout, dplm = plm_holmes(LMAX, np.cos(th))
    else:
        # use precomputed plms to improve computational speed
        # or to use a different recursion relation for polynomials
        plmout = PLM

    # Multiply plms by integration factors [sin(theta)*dtheta*dphi]
    # truncate plms to maximum spherical harmonic order if MMAX < LMAX
    m = np.arange(MMAX + 1)
    for j in range(0, nlat):
        plm[:, m, j] = plmout[:, m, j] * int_fact[j]

    # Initializing preliminary spherical harmonic matrices
    yclm = np.zeros((LMAX + 1, MMAX + 1)).astype(np.float64)
    yslm = np.zeros((LMAX + 1, MMAX + 1)).astype(np.float64)
    # Initializing output spherical harmonic matrices
    clm = np.zeros((LMAX + 1, MMAX + 1)).astype(np.float64)
    slm = np.zeros((LMAX + 1, MMAX + 1)).astype(np.float64)
    # Multiplying gridded data with sin/cos of m#phis (output [m,theta])
    # This will sum through all phis in the dot product
    dcos1 = ccos * data
    dsin1 = ssin * data
    for l in range(0, LMAX + 1):
        mm = np.min([MMAX, l])  # truncate to MMAX if specified (if l > MMAX)
        m = np.arange(0, mm + 1)  # mm+1 elements between 0 and mm
        # Summing product of plms and data over all latitudes
        yclm[l, m] = np.sum(plm[l, m, :] * dcos1[m, :], axis=1)
        yslm[l, m] = np.sum(plm[l, m, :] * dsin1[m, :], axis=1)
        # convert to output normalization (4-pi normalized harmonics)
        clm[l, m] = coeff * yclm[l, m]
        slm[l, m] = coeff * yslm[l, m]

    # return the output spherical harmonics object
    return clm, slm
