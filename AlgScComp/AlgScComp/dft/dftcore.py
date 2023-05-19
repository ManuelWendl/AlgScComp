"""
Discrete Fourier Transform 

This module contains the following algorithms:

dft(f) 
idft(F)
fft_rek(f)
ifft_rek(F)

i = inverse
_. = type of implementation

"""

from . import dfthelper as dfth
from . import dftextension as dfte

import cmath
import math

__all__ = ["dft", "idft", "fft_rek", "ifft_rek", "fft_itt_v1", "ifft_itt_v1"]


def butterfly_v1(X, inverse):
    # X requires to be a power of 2 and sorted by fftshift
    N = len(X)
    p = int(math.log2(N))  # size(N) = 2^p

    # decide if inverse or not
    b = 1 if inverse == 1 else -1

    for L in [2**l for l in range(1, p + 1)]:
        for k in range(0, N, L):
            for j in range(0, int(L / 2)):
                wz = cmath.exp(b * 1j * 2 * cmath.pi * j / L) * X[k + j + int(L / 2)]
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
        If the input list is empty.
    """
    N = len(X)

    b = 1 if inverse else -1

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
            wz = cmath.exp(b * 1j * 2 * cmath.pi * k / N) * z[k]
            x[k] = y[k] + wz
            x[k + int(N / 2)] = y[k] - wz
    else:
        x = X

    return x


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
            [f[n] * cmath.exp(-1j * 2 * cmath.pi * n * k / N) for n in range(0, N)]
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
            [F[n] * cmath.exp(1j * 2 * cmath.pi * n * k / N) for n in range(0, N)]
        )
        for k in range(0, N)
    ]

    return [dfth.get_norm_inv*f_i for f_i in f]


def fft_rek(f: list, norm: str = 'fwd') -> list:
    """
    Computation of the 1D Discrete Fourier Transform.

    Computation based on the recursive FFT devide and conquere scheme. Complexity of the recursive scheme is given by O(N log(N)).

    Parameters
    ----------
    f: list
        Input can be complex. size of f is required to be 2**n.
    norm: str
        String identifier of the norm type. Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.
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
        If the norm string identifier is unknown.
    """

    if math.log(len(f), 2) % 1 != 0 or len(f) < 1:
        raise ValueError(
            "Invalid input list size. Input list size is expected to have len 2**n and > 0"
        )

    F = fft_rek_recurs(f, 0)

    return [dfth.get_norm_fwd(len(F),norm) * F_i for F_i in F]


def ifft_rek(F: list, norm: str = 'fwd') -> list:
    """
    Computation of the 1D inverse Discrete Fourier Transform.

    Computation based on the recursive FFT devide and conquere scheme. Complexity of the recursive scheme is given by O(N log(N)).

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
        If the input list has a different size than 2**n or is empty.
    ValueError:
        If the norm string identifier is unknown.
    """

    if math.log(len(F), 2) % 1 != 0 or len(F) < 1:
        raise ValueError(
            "Invalid input list size. Input list size is expected to have len 2**n and > 0"
        )

    f = fft_rek_recurs(F,1)

    return [dfth.get_norm_inv(len(f),norm) * f_i for f_i in f]


def fft_itt_v1(f, norm: str = 'fwd') -> list:
    F = butterfly_v1(dfte.fftshift(f), 0)
    return [dfth.get_norm_fwd(len(F),norm)*F_i for F_i in F]


def ifft_itt_v1(F, norm: str = 'fwd') -> list:
    f = butterfly_v1(dfte.fftshift(F), 1)
    return [dfth.get_norm_inv(len(F),norm)*F_i for F_i in F]


f = [1, 2, 3, 4, 5, 6, 7, 8]
print(fft_rek(f))
print('---------')
print(f)
print('---------')
print(dft(f))
print('---------')
print(f)
print('---------')
print(fft_itt_v1(f))
print('---------')
print(f)
