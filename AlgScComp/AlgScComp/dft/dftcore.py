"""
Discrete Fourier Transform 

This module contains the following algorithms:

dft(f) 
idft(F)
fftf)
ifft(F)

- dft and idft are simple implementations in form of a matrix vector multiplication of runtime O(N^2)
- fft and ifft are divide and conquere alegorithms with different types of implemenatations.

"""

from . import dfthelper as dfth
from . import dftextension as dfte

import cmath
import math

'''
Numpy is only imported for vecorised impelemntation
Other functionalities are not used!
Numpy is based on C and using SIMD.
'''
import numpy as np


__all__ = ["dft", "idft", "fft", "ifft"]

'''
Helper Functions:

These functions are called by the callable functions for parts of the calcualtions and schemes. 
'''

def butterfly_v2(X: list, inverse: bool) -> list:
    '''
    Iterative implementation of the butterfly scheme:

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

    return [dfth.get_norm_fwd(N, norm)*F_i for F_i in F]


def idft(F: list, norm: str = 'fwd') -> list:
    """
    Computation of the 1D Discrete Fourier Transform.

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

    return [dfth.get_norm_inv(N,norm)*f_i for f_i in f]


def fft(f: list, norm: str = 'fwd', vers: str = 'vec') -> list:
    """
    Computation of the 1D Fast Fourier Transform.

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

    if  vers == 'v1':
        F = butterfly_v1(dfte.fftshift(f), inverse)
    elif vers == 'vec':
        F = butterfly_vec(dfte.fftshift(f), inverse)   
    elif vers == 'v2':
        F = butterfly_v2(dfte.fftshift(f), inverse)
    elif vers == 'rec':
        F = fft_rek_recurs(f, inverse)
    else:
        raise ValueError('Invalid string identifier for vers {vers}. vers is required to be (1.) v1, (2.) vec, (3.) v2 or (4.) rec')

    return [dfth.get_norm_fwd(len(F),norm) * F_i for F_i in F]


def ifft(F: list, norm: str = 'fwd', vers: str = 'vec') -> list:
    """
    Computation of the 1D inverse Fast Fourier Transform.

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

    if  vers == 'v1':
        f = butterfly_v1(dfte.fftshift(F), inverse)
    elif vers == 'vec':
        f = butterfly_vec(dfte.fftshift(F), inverse)   
    elif vers == 'v2':
        f = butterfly_v2(dfte.fftshift(F), inverse)
    elif vers == 'rec':
        f = fft_rek_recurs(F, inverse)
    else:
        raise ValueError('Invalid string identifier for vers {vers}. vers is required to be (1.) v1, (2.) vec, (3.) v2 or (4.) rec')

    return [dfth.get_norm_inv(len(f),norm) * f_i for f_i in f]


