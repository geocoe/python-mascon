import numpy as np
from os import path,makedirs
from pathlib import Path
from urllib.request import urlretrieve
from time import sleep
import sys
import struct
from scipy.linalg import block_diag


def filterSH(W, cilm, cilm_std=None):
    '''
    Filter spherical harmonic coefficients with a block diagonal fitler matrix.

    Usage:
    cilm_filter = filterSH(W,cilm)
    cilm_filter,cilm_std_filter = filterSH(W,cilm,cilm_std)

    Inputs:
    W -> [dic]: Dictionary containing the filter matrix (read by read_BIN.py)
    cilm -> [float 3d array] Spherical harmonic coefficients in matrix form. cilm = clm for i = 0; cilm = slm for i = 1

    Parameters:
    cilm_std -> [float 3d array] standard deviation for the spherical harmonic coefficients

    Outputs:
    cilm_filter -> [float 3d array] Filtered spherical harmonic coefficients
    cilm_std_filter -> [float 3d array] standard deviation for the filtered spherical harmonic coefficients

    Notice:
    This program is translated from the matlab/octave source code filterSH.m written by Roelof Rietbroek 2016.
    For more information, please refer to https://github.com/strawpants/GRACE-filter
    '''
    # Extract filter matrix
    # Maximum degree of the input coefficients
    lmax = cilm.shape[1] - 1

    # Extract the minimum and maximum degree supported by the filter matrix
    lmaxfilt, lminfilt = W['Lmax'], W['Lmin']

    # Determine the output maximum degree (limited by either the filter or input data)
    lmaxout = min(lmax, lmaxfilt)

    # Reserve space for output (will have same size as input) and set to zero
    cilm_filter = np.zeros_like(cilm)
    cilm_std_filter = np.zeros_like(cilm_std)

    # Loop parameter indicating the previous block number and the end position in the packed matrix of the previous block
    lastblckind, lastindex = 0, 0

    # loop over the available blocks
    for iblk in range(W['Nblocks']):
        # Get the degree of the block from the block index
        degree = (iblk + 1) // 2

        # Break loop if the degrees of the block are larger than the degrees of the input
        if degree > lmaxout: break
        trig = (iblk + int(iblk > 0) + 1) % 2

        # Compute the size of the side of the stored block
        sz = W['blockind'][iblk] - lastblckind

        # Initialize the filter order block to a unit diagonal matrix
        blockn = np.identity(lmaxfilt + 1 - degree)

        # Minimum (stored) degree for this particular block (may be limited by the mininum degree supported by the filter)
        lminblk = max(lminfilt, degree)

        shift = lminblk - degree
        # unpack the stored filterblock (vector) in a fully occupied order block matrix
        blockn[shift:, shift:] = W['pack1'][lastindex:lastindex + sz ** 2].reshape(sz, sz).T

        # Filter the input coefficients (this is in fact just a matrix vector multiplication)
        if trig:
            cilm_filter[0, degree:lmaxout + 1, degree] = np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree],
                                                                cilm[0, degree:lmaxout + 1, degree])
        else:
            cilm_filter[1, degree:lmaxout + 1, degree] = np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree],
                                                                cilm[1, degree:lmaxout + 1, degree])

        if cilm_std is not None:
            if trig:
                cilm_std_filter[0, degree:lmaxout + 1, degree] = np.sqrt(
                    np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree] ** 2,
                           cilm_std[0, degree:lmaxout + 1, degree] ** 2))
            else:
                cilm_std_filter[1, degree:lmaxout + 1, degree] = np.sqrt(
                    np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree] ** 2,
                           cilm_std[1, degree:lmaxout + 1, degree] ** 2))

        # Prepare the loop variables for next block
        lastblckind = W['blockind'][iblk]
        lastindex = lastindex + sz ** 2

    if cilm_std is None:
        return cilm_filter
    else:
        return cilm_filter, cilm_std_filter
def read_BIN(file, mode='packed'):
    '''
    Read the binary file containing symmetric/full or block diagonal matrices and associated vectors and parameters.

    Usage:
    dat = read_BIN(file)
    dat = read_BIN(file, mode = 'packed')
    dat = read_BIN(file, mode = 'full')

    Inputs:
    file -> [str] Input file

    Parameters:
    mode -> [optional, str, default = 'packed'] Available options are 'packed' or 'full' form for the filter matrices.
    If 'packed', the matrix remains in packed form (dat['pack1'] field). If 'full', the matrix expands to its full form (dat['mat1'] field).
    Warning: the 'full' option may cause excessive RAM memory use with large matrices.

    Outputs:
    dat -> [dic]: Dictionary with the file content

    Notice:
    This program is translated from the matlab/octave source code read_BIN.m written by Roelof Rietbroek 2016.
    For more information, please refer to https://github.com/strawpants/GRACE-filter
    '''
    if mode == 'packed':
        unpack = False
    elif mode == 'full':
        unpack = True  # unpack matrix in full size
    else:
        raise Exception("Only 'packed' or 'full' are avaliable.")

    # ckeck endian
    endian = sys.byteorder

    if endian == 'little':
        # open the binary file in little endian
        f = open(file, 'rb')
    else:
        raise Exception('The endian of the binary file is little, but the endian of OS is big.')

    dat = {}
    # read the data version and type from the binary file
    dat['version'] = f.read(8).decode().strip()
    dat['type'] = f.read(8).decode()
    dat['descr'] = f.read(80).decode().strip()

    for key in ['nints', 'ndbls', 'nval1', 'nval2']:
        dat[key] = struct.unpack('<I', f.read(4))[0]

    for key in ['pval1', 'pval2']:
        dat[key] = struct.unpack('<I', f.read(4))[0]

    dat['nvec'], dat['pval2'] = 0, 1
    dat['nread'], dat['nval2'] = 0, dat['nval1']

    # read additional nblocks parameter
    nblocks = struct.unpack('<i', f.read(4))[0]

    lists = f.read(dat['nints'] * 24).decode().split()
    for element in lists:
        dat[element] = struct.unpack('<i', f.read(4))[0]

    lists = f.read(dat['ndbls'] * 24).decode().replace(':', '').split()
    for element in lists:
        dat[element] = struct.unpack('<d', f.read(8))[0]

    # side description meta data
    lists = f.read(dat['nval1'] * 24).decode()
    dat['side1_d'] = [(lists[i:i + 24]).replace('         ', '') for i in range(0, len(lists), 24)]

    # type specific meta data
    dat['blockind'] = np.array(struct.unpack('<' + str(nblocks) + 'i', f.read(4 * nblocks)))

    dat['side2_d'] = dat['side1_d']

    # read matrix data
    npack1 = dat['pval1'] * dat['pval2']
    dat['pack1'] = np.array(struct.unpack('<' + str(npack1) + 'd', f.read(8 * npack1)))

    f.close()  # close file

    if not unpack: return dat

    sz = dat['blockind'][0]
    dat['mat1'] = dat['pack1'][:sz ** 2].reshape(sz, sz).T

    shift1 = shift2 = sz ** 2

    for i in range(1, nblocks):
        sz = dat['blockind'][i] - dat['blockind'][i - 1]
        shift2 = shift1 + sz ** 2
        dat['mat1'] = block_diag(dat['mat1'], dat['pack1'][shift1:shift2].reshape(sz, sz).T)
        shift1 = shift2
    del dat['pack1']

    return dat
def filter_ddk(filter_type, shc, direc, shc_std=None):
    '''
    DDK filter used to attenuate noise described as striping patterns in GRACE GSM data.
    According to the filtering strength, there are 8 kinds of ddk filter.
    From DDK1 to DDK8, the filtering effect gradually weakens.

    Usage:
    filt_SHC = filt_DDK('DDK3',SHC)
    filt_SHC,filt_SHC_std = filt_DDK('DDK4',SHC,SHC_std)

    Inputs:
    filt_type -> [str] Types of DDK filter. Available options are 'DDK1' to 'DDK8'.
    SHC -> [float 3d/4d array] Fully normalized spherical harmonics coefficients(Stokes coefficients). The dimension of SHC can either be 3d or 4d.
    If 3d, its shape is (i,l,m), where i = 0 for Clm and i = 1 for Slm. If 4d, its shape is (k,i,l,m).

    Parameters:
    SHC_std -> [optional, float 3d/4d array, default = None] Standard deviation for SHC.
    If None, the standard deviation for the filtered SHC will not be estimated, and only the filtered SHC is returned.

    Outputs:
    filt_SHC -> [float 3d/4d array] filtered SHC
    filt_SHC_std -> [float 3d/4d array] standard deviation for filtered SHC

    For more information, please refer to https://github.com/strawpants/GRACE-filter
    '''
    filter_shc = np.zeros_like(shc)
    filter_shc_std = np.zeros_like(shc_std)

    # Download the DDK filter matrices
    ddk_types = ['1d14', '1d13', '1d12', '5d11', '1d11', '5d10', '1d10', '5d9']
    urldir = 'https://raw.githubusercontent.com/strawpants/GRACE-filter/master/data/DDK/'

    if not path.exists(direc):
        makedirs(direc)
        for ddk_type in ddk_types:
            ddk_bin = 'Wbd_2-120.a_' + ddk_type + 'p_4'
            url = urldir + ddk_bin
            for idownload in range(3):
                print('Downloading the DDK filter matrix ... ' + ddk_bin, end=' ... ')
                try:
                    urlretrieve(url, direc + ddk_bin)
                    print('Transfer completed')
                    break
                except:
                    print('Transfer failed, try downloading again.')
                sleep(30)  # Pause for 30 seconds

    # read the filter matrix

    if filter_type == 'DDK1':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_1d14p_4')
    elif filter_type == 'DDK2':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_1d13p_4')
    elif filter_type == 'DDK3':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_1d12p_4')
    elif filter_type == 'DDK4':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_5d11p_4')
    elif filter_type == 'DDK5':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_1d11p_4')
    elif filter_type == 'DDK6':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_5d10p_4')
    elif filter_type == 'DDK7':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_1d10p_4')
    elif filter_type == 'DDK8':
        Wbd = read_BIN(direc + 'Wbd_2-120.a_5d9p_4')
    else:
        raise Exception('Currently, only DDK1~DDK8 are feasible.')

    if shc.ndim == 4:

        if shc_std is None:
            for i in range(shc.shape[0]):
                filter_shc[i] = filterSH(Wbd, shc[i])
            return filter_shc
        else:
            for i in range(shc.shape[0]):
                filter_shc[i], filter_shc_std[i] = filterSH(Wbd, shc[i], shc_std[i])
            return filter_shc, filter_shc_std

    elif shc.ndim == 3:

        if shc_std is None:
            filter_shc = filterSH(Wbd, shc)
            return filter_shc
        else:
            filter_shc, filter_shc_std = filterSH(Wbd, shc, shc_std)
            return filter_shc, filter_shc_std
    else:
        raise Exception('Dimension of the SHC data is not correct. It should be 3 or 4.')