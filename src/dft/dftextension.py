'''
Helper Functions of the Discrete Fourier Module

'''
import cmath
import math

__all__ = ['fftshift','dftMatrix']

def dftMatrix(N:int = 256) -> list:
    '''
    Computation of the DFT Matrix

    Parameters
    ----------
    N: int
        Size of the quadratic DFT matrix. If N is not given N = 256.

    Returns 
    -------
    out: complex matrix
        complex dft matrix of size N^2

    Raises
    ------
    IndexError: 
        If the input dimension is invalid

    '''

    if N < 1: raise ValueError('Invalid size given.')

    dftMat = [[cmath.exp(1J*2*cmath.pi*n*k/N) for n in range(0,N)]for k in range (0,N)]

    return dftMat


def fftshift(X:list) -> list:
    '''
    Perform Fast Fourier Shift.

    The shift algorithm is based on bit inverting, which is equivalent to sorting.

    Parameters
    ----------
    X: list
        list can be complex. Is required to have length 2**n

    Returns 
    -------
    out: list
        In case of a complex input out also compex. Shifted 

    Raises
    ------
    IndexError: 
        If the input 

    '''

    if (math.log(len(X),2)%1 != 0 or len(X)<1): raise ValueError('Invalid input list size. Input list size is expected to have len 2**n')

    p = int(math.log(len(X),2)) 

    for n in range(0,len(X)):
        j = 0; m = n
        for i in range(0,p):
            j = int(2*j + m%2); m = int(m/2)
        if (j>n):
            h = X[j]; X[j] = X[n]; X[n] = h
    
    return X