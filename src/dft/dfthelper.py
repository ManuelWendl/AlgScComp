'''
Helper functions of the Discrete Fourier Transform module

'''


import math

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