"""
Discrete Fourier Transform Core
===============================

This module contains the following algorithms:

dft(f) 
idft(F)
fft(f)
ifft(F)
rfft(f)

- dft and idft are simple implementations in form of a matrix vector multiplication of runtime O(N^2)
- fft and ifft are divide and conquere alegorithms with different types of implemenatations.

"""

from . import dfthelper as dfth
from . import dftextension as dfte

import math
import time

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import ctypes as ct

dftc = ct.CDLL(dir_path+'/dftc.so')
dftc.connect()


'''
Numpy is only imported for vecorised impelemntation
Other functionalities are not used!
Numpy is based on C and using SIMD.
'''
import numpy as np


__all__ = ["dft", "idft", "fft", "ifft", "rfft"]

'''
Helper Functions:

These functions are called by the callable functions for parts of the calcualtions and schemes. 
'''

def butterfly_v2(X: list, inverse: bool) -> list:
    '''
    Iterative implementation of the butterfly scheme:
    =================================================

    Version 2: j - outer loop and k - inner loop.No stride 1 anymore, such that no vectorised implementation is possible. 
    But as a benefit the w complex unit roots are calculated in the outer for loop. 

    Parameters
    ----------
    X: list
        Input can be complex.
    inverse: bool
        Identifying if inverse scheme is required. If inverse = 0: not inverse. If inverse = 1: inverse scheme is applied.

    Returns
    -------
    x: complex list
        Complex Fourier coefficients in case inverse == 0, data points from Fourier Coefficients in case inverse == 1

    Raises
    ------
    IndexError:
        If the length of the input list is no power of 2 or empty.

    '''
    # X requires to be a power of 2 and sorted by fftshift
    N =  len(X)
    p = int(math.log2(N)) # size(N) = 2^p

    if p % 1 != 0 or N < 1:
        raise ValueError(
            "Invalid input list size. Input list size is expected to have len 2**n and > 0"
        )

    for  L in [2**l for l in range(1, p + 1)]:
        for j in range(0,int(L/2)):
            w = dfth.omega(j,L,inverse)
            for k in range(0,N,L):
                wz = w*X[k+j+int(L/2)]
                X[k+j+int(L/2)] = X[k+j] - wz
                X[k+j] = X[k+j] + wz
    return X


def butterfly_vec(X: list, inverse: bool) -> list:
    '''
    Iterative implementation of the butterfly scheme:
    =================================================

    Version 1 vectorised: k - outer loop and j - inner loop.
    Stride 1 allows us to implement this vectorised using SIMD instructions in the backround, instead of elemnt wise operations over the for loop.
    For the vectorised implementation the numpy module is used. 
    Only the vectorisation feature is used, see disclamer in the import section. 

    Parameters
    ----------
    X: list
        Input can be complex.
    inverse: bool
        Identifying if inverse scheme is required. If inverse = 0: not inverse. If inverse = 1: inverse scheme is applied.

    Returns
    -------
    x: complex list
        Complex Fourier coefficients in case inverse == 0, data points from Fourier Coefficients in case inverse == 1

    Raises
    ------
    IndexError:
        If the length of the input list is no power of 2 or empty.

    '''
    X = np.array(X,np.complex_)
    N =  len(X)
    p = int(math.log2(N)) # size(N) = 2^p

    if p % 1 != 0 or N < 1:
        raise ValueError(
            "Invalid input list size. Input list size is expected to have len 2**n and > 0"
        )
    
    # decide if inverse or not
    b = 1 if inverse == 1 else -1

    # define SIMD_LENGTH
    SIMD_LENGTH = 8

    for  L in [2**l for l in range(1, p + 1)]:
        d = min([SIMD_LENGTH,int(L/2)])
        for k in range(0,N,L):
            for j in range(0,int(L/2),SIMD_LENGTH):
                kjStart = k+j
                kjEnde = k+j+d
                wz = np.multiply(np.exp(b*1J*2*np.pi*np.arange(kjStart-k,kjEnde-k)/L),X[kjStart+np.int_(L/2):kjEnde+np.int_(L/2)])
                X[kjStart+int(L/2):kjEnde+int(L/2)] = np.subtract(X[kjStart:kjEnde],wz)
                X[kjStart:kjEnde] = np.add(X[kjStart:kjEnde],wz)
    return X


def butterfly_v1(X: list, inverse: bool) -> list:
    '''
    Iterative impllementation of the butterfly scheme:
    ==================================================

    Version 1: k - outer loop and j - inner loop.

    Parameters
    ----------
    X: list
        Input can be complex.
    inverse: bool
        Identifying if inverse scheme is required. If inverse = 0: not inverse. If inverse = 1: inverse scheme is applied.

    Returns
    -------
    x: complex list
        Complex Fourier coefficients in case inverse == 0, data points from Fourier Coefficients in case inverse == 1

    Raises
    ------
    IndexError:
        If the length of the input list is no power of 2 or empty.

    '''
    # X requires to be a power of 2 and sorted by fftshift
    N = len(X)

    p = int(math.log2(N))  # size(N) = 2^p

    if p % 1 != 0 or N < 1:
        raise ValueError(
            "Invalid input list size. Input list size is expected to have len 2**n and > 0"
        )

    for L in [2**l for l in range(1, p + 1)]:
        for k in range(0, N, L):
            for j in range(0, int(L / 2)):
                wz = dfth.omega(j,L,inverse) * X[k + j + int(L / 2)]
                X[k + j + int(L / 2)] = X[k + j] - wz
                X[k + j] = X[k + j] + wz
    return X


def fft_rek_recurs(X: list, inverse: bool) -> list:
    """
    Recursive call of the recursive implementation of the Fast Fourier Transform
    ============================================================================
    
    Parameters
    ----------
    X: list
        Input can be complex.
    inverse: bool
        Identifying if inverse scheme is required. If inverse = 0: not inverse. If inverse = 1: inverse scheme is applied.

    Returns
    -------
    x: complex list
        Complex Fourier coefficients in case inverse == 0, data points from Fourier Coefficients in case inverse == 1

    Raises
    ------
    IndexError:
        If the length of the input list is no power of 2 or empty.
    """
    N = len(X)

    if math.log(N, 2) % 1 != 0 or N < 1:
        raise ValueError(
            "Invalid input list size. Input list size is expected to have len 2**n and > 0"
        )

    if N > 1:
        Y = [complex()] * int(N / 2)
        Z = [complex()] * int(N / 2)

        for i in range(0, int(N / 2)):
            Y[i] = X[2 * i]
            Z[i] = X[2 * i + 1]

        y = fft_rek_recurs(Y, inverse)
        z = fft_rek_recurs(Z, inverse)

        x = [complex()] * N
        for k in range(0, int(N / 2)):
            wz = dfth.omega(k,N,inverse) * z[k]
            x[k] = y[k] + wz
            x[k + int(N / 2)] = y[k] - wz
    else:
        x = X

    return x

'''
Callable Functions:

These functions are available from outside the module.
'''

def dft(f: list, norm: str = 'fwd') -> list:
    """
    Computation of the 1D Discrete Fourier Transform.
    =================================================

    Basic implementation with O(N^2) runtime. Equivalent to the matrix vector multiplication.

    Parameters
    ----------
    f: list
        Input can be complex.
    norm: str
        String identifier of the norm type. Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.
        Default value is given (1.) 'fwd'
                
    Returns
    -------
    out: complex list
        Complex Fourier coefficients.

    Raises
    ------
    IndexError:
        If the input list is empty.
    ValueError:
        If the norm string identifier is unknown.

    """

    N = len(f)

    if N < 1:
        raise ValueError("Empty list given. Invalid number of data points.")

    F = [
        sum(
            [f[n] * dfth.omega(k*n,N,0) for n in range(0, N)]
        )
        for k in range(0, N)
    ]

    return [dfth.cround(dfth.get_norm_fwd(N, norm)*F_i,7) for F_i in F]


def idft(F: list, norm: str = 'fwd') -> list:
    """
    Computation of the 1D Discrete Fourier Transform.
    =================================================

    Basic implementation with O(N^2) runtime. Equivalent to the matrix vector multiplication.

    Parameters
    ----------
    F: complex list
        Complex Fourier coefficients. Size of F is required to be 2**n.

    Returns
    -------
    f: complex list
        Data points of inverse transformed DFT coefficients.
    norm: str
        String identifier of the norm type. Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.
        Default value is given (1.) 'fwd'

        
    Raises
    ------
    IndexError:
        If the input list is empty.
    ValueError:
        If the norm string identifier is unknown.

    """

    N = len(F)

    if N < 1:
        raise ValueError("Empty list given. Invalid number of data points.")

    f = [
        sum(
            [F[n] * dfth.omega(k*n,N,1) for n in range(0, N)]
        )
        for k in range(0, N)
    ]

    return [dfth.cround(dfth.get_norm_inv(N,norm)*f_i,7) for f_i in f]


def fft(f: list, norm: str = 'fwd', vers: str = 'vec', lang: str = 'c') -> list:
    """
    Computation of the 1D Fast Fourier Transform.
    ============================================

    Computation based on diffreent FFT devide and conquere schemes. 
    The Complexity of the devide and conquere schemes is given by O(N log(N)).
    Different versions can be addressed by the function parameter vers, which offers the different computation schemes. 

    Parameters
    ----------
    f: list
        Input can be complex. size of f is required to be 2**n.
    vers: str
        String identifier of the computation scheme. 
        Possible options are (1.) 'v1', (2.) 'vec', (3.) 'v2' and (4.) 'rec'
        Default value is given (2.) 'vec'
    norm: str
        String identifier of the norm type. 
        Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.
        Default value is given (1.) 'fwd'
    lang: str
        String identifier if the language specification.
        Possible options are (1.) 'c', (2.) 'py'.
        Default value is given (1.) 'c' (except for recursive).
        'c' implementation is faster than the python implementation. 

    Returns
    -------
    F: complex list
        Complex Fourier coefficients.

    Raises
    ------
    IndexError:
        If the input list has a different size than 2**n or is empty.
    ValueError:
        If the vers string identifier is unknown.
    ValueError:
        If the norm string identifier is unknown.
    ValueError:
        If the lang string identifier is unknown.
        
    Notes
    -----
    v1: Itterative scheme with k outer for loop and j inner for loop. Consecutive Stride 1 access to data from array (suitable for vectorisation). 
        w is computed in innermost loop.

    vec: Itterative scheme analog to v1 but in a vectorised implementation. For this function the SIMD implemenatation of numpy is used. 
        No additional features than array addiatioan and multiplication are taken from numpy.
    
    v2: Itterative scheme with j outer for loop and k inner for loop. No consecutive data access (no vectorised implementation possible).
        w is computed in the outer for loop. Complex computation is expensive and therefore compuatational advantage over v1 (but not vec).

    rec: Recursive computation scheme. The recursive implementation shows good memeory usage and therefore short data access times.
    """

    inverse = False

    fac = dfth.get_norm_fwd(len(f),norm)

    if  vers == 'v1':
        if lang == 'c':
            class ComplexC(ct.Structure):
                _fields_ = [("real", ct.POINTER(ct.c_float)),
                ("imag",ct.POINTER(ct.c_float))]

            f_real, f_imag = dfth.splitComplex(f)

            freal = ((ct.c_float) * len(f))(*f_real)
            fimag = ((ct.c_float) * len(f))(*f_imag)

            dftc.butterfly_v1.restype = ct.POINTER(ComplexC)

            t = time.time()
            Fp = dftc.butterfly_v1(ComplexC(freal,fimag), ct.c_bool(inverse), ct.c_int(len(f)))
            elapsed = time.time()-t
            print('V1 Elapsed Time:',elapsed)

            F = [dfth.cround(fac*Fp.contents.real[i],7) + 1J * dfth.cround(fac*Fp.contents.imag[i],7) for i in range (0,len(f))]

            dftc.freeMemory(Fp)

        elif lang == 'py':
            Fn = butterfly_v1(dfte.fftshift(f), inverse)
            F = [dfth.cround(fac * F_i,7) for F_i in Fn]
        else:
            raise ValueError('Invalid string identifier for lang {lang}. lang is required to be (1.) c, (2.) py')

    elif vers == 'vec':
        if lang == 'c':
            class ComplexC(ct.Structure):
                _fields_ = [("real", ct.POINTER(ct.c_float)),
                ("imag",ct.POINTER(ct.c_float))]

            f_real, f_imag = dfth.splitComplex(f)

            freal = ((ct.c_float) * len(f))(*f_real)
            fimag = ((ct.c_float) * len(f))(*f_imag)

            dftc.butterfly_vec.restype = ct.POINTER(ComplexC)
            t = time.time()
            Fp = dftc.butterfly_vec(ComplexC(freal,fimag), ct.c_bool(inverse), ct.c_int(len(f)))
            elapsed = time.time()-t
            print('Vec Elapsed Time:',elapsed)
            F = [dfth.cround(fac*Fp.contents.real[i],7) + 1J * dfth.cround(fac*Fp.contents.imag[i],7) for i in range (0,len(f))]

            dftc.freeMemory(Fp)

        elif lang == 'py':   
            Fn = butterfly_vec(dfte.fftshift(f), inverse)  
            F = [dfth.cround(fac * F_i,7) for F_i in Fn]
        else: 
            raise ValueError('Invalid string identifier for lang {lang}. lang is required to be (1.) c, (2.) py')
         
    elif vers == 'v2':
        if lang == 'c':
            class ComplexC(ct.Structure):
                _fields_ = [("real", ct.POINTER(ct.c_float)),
                ("imag",ct.POINTER(ct.c_float))]

            f_real, f_imag = dfth.splitComplex(f)

            freal = ((ct.c_float) * len(f))(*f_real)
            fimag = ((ct.c_float) * len(f))(*f_imag)

            dftc.butterfly_v2.restype = ct.POINTER(ComplexC)

            t = time.time()

            Fp = dftc.butterfly_v2(ComplexC(freal,fimag), ct.c_bool(inverse), ct.c_int(len(f)))

            elapsed = time.time()-t
            print('V2 Elapsed Time:',elapsed)

            F = [dfth.cround(fac*Fp.contents.real[i],7) + 1J * dfth.cround(fac*Fp.contents.imag[i],7) for i in range (0,len(f))]

            dftc.freeMemory(Fp)

        elif lang == 'py':
            Fn = butterfly_v2(dfte.fftshift(f), inverse)
            F = [dfth.cround(fac * F_i,7) for F_i in Fn]
        else:
            raise ValueError('Invalid string identifier for lang {lang}. lang is required to be (1.) c, (2.) py')
        
    elif vers == 'rec':
        if lang == 'c':
            print('Recursive scheme is not available in C. Computed in Python.')
        Fn = fft_rek_recurs(f, inverse)
        F = [dfth.cround(fac * F_i,7) for F_i in Fn]
    else:
        raise ValueError('Invalid string identifier for vers {vers}. vers is required to be (1.) v1, (2.) vec, (3.) v2 or (4.) rec')

    return F


def ifft(F: list, norm: str = 'fwd', vers: str = 'vec', lang: str = 'c') -> list:
    """
    Computation of the 1D inverse Fast Fourier Transform.
    =====================================================

    Computation based on diffreent FFT devide and conquere schemes. 
    The Complexity of the devide and conquere schemes is given by O(N log(N)).
    Different versions can be addressed by the function parameter vers, which offers the different computation schemes. 

    Parameters
    ----------
    F: complex list
        Complex Fourier coefficients. Size of F is required to be 2**n.
    vers: str
        String identifier of the computation scheme. 
        Possible options are (1.) 'v1', (2.) 'vec', (3.) 'v2' and (4.) 'rec'
        Default value is given (2.) 'vec'
    norm: str
        String identifier of the norm type. 
        Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.
        Default value is given (1.) 'fwd'
    lang: str
        String identifier if the language specification.
        Possible options are (1.) 'c', (2.) 'py'.
        Default value is given (1.) 'c' (except for recursive).
        'c' implementation is faster than the python implementation. 

    Returns
    -------
    f: complex list
        Data points of inverse transformed DFT coefficients.

    Raises
    ------
    IndexError:
        If the input list has a different size than 2**n or is empty.
    ValueError:
        If the norm string identifier is unknown.
    ValueError:
        If the norm string identifier is unknown.
    ValueError:
        If the lang string identifier is unknown.

    Notes
    -----
    v1: Itterative scheme with k outer for loop and j inner for loop. Consecutive Stride 1 access to data from array (suitable for vectorisation). 
        w is computed in innermost loop.

    vec: Itterative scheme analog to v1 but in a vectorised implementation. For this function the SIMD implemenatation of numpy is used. 
        No additional features than array addiatioan and multiplication are taken from numpy.
    
    v2: Itterative scheme with j outer for loop and k inner for loop. No consecutive data access (no vectorised implementation possible).
        w is computed in the outer for loop. Complex computation is expensive and therefore compuatational advantage over v1 (but not vec).

    rec: Recursive computation scheme. The recursive implementation shows good memeory usage and therefore short data access times.
    """

    inverse = True
    fac = dfth.get_norm_inv(len(F),norm)

    if  vers == 'v1':
        if lang == 'c':
            class ComplexC(ct.Structure):
                _fields_ = [("real", ct.POINTER(ct.c_float)),
                ("imag",ct.POINTER(ct.c_float))]

            f_real, f_imag = dfth.splitComplex(F)

            freal = ((ct.c_float) * len(F))(*f_real)
            fimag = ((ct.c_float) * len(F))(*f_imag)

            dftc.butterfly_v1.restype = ct.POINTER(ComplexC)

            Fp = dftc.butterfly_v1(ComplexC(freal,fimag), ct.c_bool(inverse), ct.c_int(len(F)))
            
            f = [dfth.cround(fac*Fp.contents.real[i],7) + 1J * dfth.cround(fac*Fp.contents.imag[i],7) for i in range (0,len(F))]

            dftc.freeMemory(Fp)

        elif lang == 'py':
            fn = butterfly_v1(dfte.fftshift(F), inverse)
            f = [dfth.cround(fac * f_i,7) for f_i in fn]
        else:
            raise ValueError('Invalid string identifier for lang {lang}. lang is required to be (1.) c, (2.) py')

    elif vers == 'vec':
        if lang == 'c':
            class ComplexC(ct.Structure):
                _fields_ = [("real", ct.POINTER(ct.c_float)),
                ("imag",ct.POINTER(ct.c_float))]

            f_real, f_imag = dfth.splitComplex(F)

            freal = ((ct.c_float) * len(F))(*f_real)
            fimag = ((ct.c_float) * len(F))(*f_imag)

            dftc.butterfly_vec.restype = ct.POINTER(ComplexC)

            Fp = dftc.butterfly_vec(ComplexC(freal,fimag), ct.c_bool(inverse), ct.c_int(len(F)))

            f = [dfth.cround(fac*Fp.contents.real[i],7) + 1J * dfth.cround(fac*Fp.contents.imag[i],7) for i in range (0,len(F))]

            dftc.freeMemory(Fp)

        elif lang == 'py':   
            fn = butterfly_vec(dfte.fftshift(F), inverse)  
            f = [dfth.cround(fac * f_i,7) for f_i in fn]
        else: 
            raise ValueError('Invalid string identifier for lang {lang}. lang is required to be (1.) c, (2.) py')
         
    elif vers == 'v2':
        if lang == 'c':
            class ComplexC(ct.Structure):
                _fields_ = [("real", ct.POINTER(ct.c_float)),
                ("imag",ct.POINTER(ct.c_float))]

            f_real, f_imag = dfth.splitComplex(F)

            freal = ((ct.c_float) * len(F))(*f_real)
            fimag = ((ct.c_float) * len(F))(*f_imag)

            dftc.butterfly_v2.restype = ct.POINTER(ComplexC)

            Fp = dftc.butterfly_v2(ComplexC(freal,fimag), ct.c_bool(inverse), ct.c_int(len(F)))

            f = [dfth.cround(fac*Fp.contents.real[i],7) + 1J * dfth.cround(fac*Fp.contents.imag[i],7) for i in range (0,len(F))]

            dftc.freeMemory(Fp)

        elif lang == 'py':
            fn = butterfly_v2(dfte.fftshift(F), inverse)
            f = [dfth.cround(fac * f_i,7) for f_i in fn]
        else:
            raise ValueError('Invalid string identifier for lang {lang}. lang is required to be (1.) c, (2.) py')
        
    elif vers == 'rec':
        if lang == 'c':
            print('Recursive scheme is not available in C. Computed in Python.')
        fn = fft_rek_recurs(F, inverse)
        f = [dfth.cround(dfth.get_norm_inv(len(fn),norm) * f_i,7) for f_i in fn]
    else:
        raise ValueError('Invalid string identifier for vers {vers}. vers is required to be (1.) v1, (2.) vec, (3.) v2 or (4.) rec')

    return f


def rfft(f: list, norm: str = 'fwd',lang: str = 'c') -> list:
    """
    Computation of the 1D Fast Fourier Transform of purely real data.
    =================================================================

    omputation of the FFT for purely real data, based on two different schemes:
    1. A 2N real data vector is compuetet by 2 * N complex FFTs
    2. Twon N real valued vactors are computet simultaneously

    Parameters
    ----------
    f: list
        Input must be real. size of f is required to be 2**n.
    norm: str
        String identifier of the norm type. 
        Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.
        Default value is given (1.) 'fwd'
    lang: str
        String identifier if the language specification.
        Possible options are (1.) 'c', (2.) 'py'.
        Default value is given (1.) 'c' (except for recursive).
        'c' implementation is faster than the python implementation. 

    Returns
    -------
    F: Complex list
        Complex Fourier coefficients.

    Raises
    ------
    IndexError:
        If the input list has a different size than 2**n or is empty.
    ValueError:
        If the dimensions of the input vector/ matrix doesnT correspond to (1,..) or (2,..)
    ValueError:
        If the norm string identifier is unknown.
    ValueError:
        If the lang string identifier is unknown.
    """
    dims = dfth.checkDimensions(f)

    fac = 1/2

    if norm == 'fwd':
        fac = 1/4
    elif norm == 'inv':
        fac = 1/2
    elif norm == 'unity':
        fac = 1/2
    else:
        raise ValueError('Invalid string identifier for norm {norm}. norm is required to be (1.) fwd, (2.) inv or (3.) unitary')
    
    if dims[0] == 1:
        
        N = dims[1]
        n = range(0,int(N/2))

        f1 = f[0:N:2]
        f2 = f[1:N:2]

        z = [f1[i] + 1J*f2[i] for i in range(0,len(f2))]
        Z = fft(z,norm,lang=lang)

        F = [None] * N

        for k in n:
            F[k] = fac*Z[k]*(1-1J*dfth.omega(k,N,False)) + fac*dfth.conj(Z[-k])*(1+1J*dfth.omega(k,N,False))
        
        n = range(int(-N/2+1),1)

        for k in n:
            F[int(N/2)-k] = dfth.conj(fac*Z[k]*(1+1J*dfth.omega(k,N,False)) + fac*dfth.conj(Z[-k])*(1-1J*dfth.omega(k,N,False)))                                   

        return F

    elif dims[0] == 2:
        
        f1 = f[0][0:dims[1]]
        f2 = f[1][0:dims[1]]
        z = [f1[i]+f2[i] * 1J for i in range(0,dims[1])]
        print(z)

        Z = fft(z,norm=norm,lang=lang)

        G1 = [Z[i]+dfth.conj(Z[-i]) for i in range(0,len(Z))]
        H1 = [Z[i]-dfth.conj(Z[-i]) for i in range(0,len(Z))]

        G = [G1i * fac  for G1i in G1]
        H = [H1i * -fac*1J for H1i in H1]

        return [G,H]


    raise ValueError("Input size of list shall not exceed 3 in first dimension. Only size (1,..) or (2,..) accepted.")
