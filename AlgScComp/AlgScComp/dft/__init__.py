"""
Discrete Fourier Transform
==========================

Module:
--------
This module contains the following algorithms related to the Discrete Fourier Transformation:

- dft(f) - Discrete Fourier Transform
- idft(F) - Inverse Disceret Fourier Transform
- fft(f) - Fast Fourier Transform 
- ifft(F) - Ineverse Fast Fourier Transform
- rfft(f) - Real Fast Fourier Transform
- fct(f) - Discrete Cosine Transform (FFT implementation)
- ifct(F) - Inverse Discrete Cosine Transform (FFT Implementation)
- qwfct(f) - Quater Wave Discrete Cosine Transform (FFT implementation)
- iqwfct(F) - Inverse Quater Wave Discrete Cosine Transform (FFT implementation) 
- fst(f) - Disceret Sine Transform (FFT implementation)
- ifst(F) - Inverse Disceret Sine Transform (FFT implementation)

Notes:
------
- dft and idft are simple implementations in form of a matrix vector multiplication of runtime O(N^2)
- fft and ifft are divide and conquere alegorithms with different types of implemenatations.
- rfft, fct, ifct, qwfct, iqwfct, fst, ifst use the fft and ifft algorithm in their transforms. 
- fft and ifft are also provided in C implementations, which are used as standard, because of faster computation times. 
"""

from .dftcore import *
from .dftextension import *

__all__ = dftcore.__all__.copy()
__all__ += dftextension.__all__
