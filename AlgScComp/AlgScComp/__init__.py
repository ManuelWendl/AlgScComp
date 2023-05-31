'''
AlgScComp - Algorithms for Scientific Computing
===============================================

Python package with implementations of algorithms in scientifc computing. 
The code is structured in different subpackages containing various topics of scientific computing.

Packages:
---------

dft: Containig various algorithms related to the Discrete Fourier Transform. 
hnm: Containing various algorithms for hierarchical numerical methods.

'''
from . import dft
from . import hnm

__all__ = ['dft','hnm']