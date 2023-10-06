'''
Helper functions of the Discrete Fourier Transform module

'''


import math
import cmath
import copy

def get_norm_fwd(N: int, norm: str = 'fwd') -> float:
    '''
    Get norm of the forward Discrete Foruier Transform. 

    In the literature there exist three differnet definitions of the dft. 
    1. Scale in the forward transform by 1/N
    2. Scale in the backward Transform by 1/N
    3. Scale in forward and backward transform by 1/sqrt(N)

    Parameters
    ----------
    N: int
        Length of the input
    norm: str
        String identifier of the norm type. Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.

    Returns
    -------
    out: float
        Scaling factor for the forward Discrete Fourier Transform.

    Raises
    ------
    ValueError:
        If the norm string identifier is unknown.
    '''

    if norm == 'fwd':
        return 1.0/N
    elif norm == 'inv':
        return 1
    elif norm == 'unity':
        return 1.0/math.sqrt(N)
    raise ValueError('Invalid string identifier for norm {norm}. norm is required to be (1.) fwd, (2.) inv or (3.) unitary')

def get_norm_inv(N: int, norm: str = 'fwd') -> float:
    '''
    Get norm of the inverse Discrete Foruier Transform. 

    In the literature there exist three differnet definitions of the dft. 
    1. Scale in the forward transform by 1/N
    2. Scale in the backward Transform by 1/N
    3. Scale in forward and backward transform by 1/sqrt(N)

    Parameters
    ----------
    N: int
        Length of the input
    norm: str
        String identifier of the norm type. Possible options are (1.) 'fwd', (2.) 'inv' and (3.) 'unitary'.

    Returns
    -------
    out: float
        Scaling factor for the inverse Discrete Fourier Transform.

    Raises
    ------
    ValueError:
        If the norm string identifier is unknown.
    '''

    if norm == 'fwd':
        return 1
    elif norm == 'inv':
        return 1.0/N
    elif norm == 'unity':
        return 1.0/math.sqrt(N)
    raise ValueError('Invalid string identifier for norm {norm}. norm is required to be (1.) fwd, (2.) inv or (3.) unitary')


def omega(j: int,L: int,inverse: bool) -> complex:
    b = 1 if inverse else -1
    return cmath.exp(b * 1j * 2 * cmath.pi * j / L)

def cround(z : complex, n: int) -> complex:
    return round(z.real,n) + 1J * round(z.imag,n)

def splitComplex(c: list):
    c_real = [c_i.real for c_i in c]
    c_imag = [c_i.imag for c_i in c]
    return c_real, c_imag

def checkDimensions(x: list) -> list:
    dim1 = len(x);
    dim2 = list();
    dim2bool = 0; # 1 one dim, 2 two dim

    for xi in x:
        if type(xi) != list:
            dim2.append(0)
            if dim2bool == 2:
                raise ValueError("List has inconsistent sizes in second dimension.")
            dim2bool = 1
        elif type(xi) == list:
            for xij in xi:
                if type(xij) == list:
                    raise ValueError("List has more than two dimension")
            dim2.append(len(xi))
            if dim2bool == 1:
                raise ValueError("List has inconsistent sizes in second dimension.")
            dim2bool = 2
    
    if sum(dim2) == 0:
        dim1 = 1
        dim2 = len(x)
    else:
        for dim2i in dim2:
            if dim2i != dim2[0]:
                raise ValueError("List has inconsistent sizes in second dimension.")
        dim2 = dim2[0]

    return [dim1, dim2]

def conj(z: complex)->complex:
    return z.conjugate()

def conjarr(l: list)->list:
    return [fi.conjugate() for fi in l]
