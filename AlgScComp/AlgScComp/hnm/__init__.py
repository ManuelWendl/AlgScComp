'''
Numerical Hierarchical Methods
==============================

Module
-------

This module contains the following algorithms related to numericalk hierarchical methods:

- plinint(xvec, yvec, x) - piecewise linear interpolation
- Classifier1Dnodal() - Classifier for 1D data based on nodal basis function representation
- quadrature1Dcomp(f,a,b,n,vers) - Numerical quadrature with composition schemes trapezoidal and simpson
- hierarchise1D(u) - Transform from nodal basis into hierarchical basis
- dehierarchise1D(v) - ransfrom for hierarchical basis into nodal basis
- Classifier1DSparse() - Classifier for 1D data based on sparse hierarchical basis function representation

'''



from . import hnmcore
from .hnmcore import *

__all__ = hnmcore.__all__