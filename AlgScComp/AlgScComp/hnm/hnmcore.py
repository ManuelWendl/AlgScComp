"""
Hierarchical Numerical Methods
==============================

This module contains the following algporithms: 

plinint(xvec, yvec, x) - piecewise linear interpolation

"""

from . import hnmhelper as hnmh

import math
import matplotlib.pyplot as plt

__all__ = ["plinint","Classifier1Dnodal","quadrature1Dcomp","hierarchise1D","dehierarchise1D","Classifier1DSparse"]

class Node:
    '''
    Node class of the Hierarchical Sparse 1D Classifier tree.
    Each node can contain one right and one left decedent node, which is memeber of the one lower layer. 
    Each node has its own scalin coefficient for approximating the 
    '''
    def __init__(self,xpoint: float,level: int,xdata: list,ydata: list):
        self.left = 0
        self.right = 0
        self.coeff = 0
        self.xpoint = xpoint
        self.level = level
        self.xdata = xdata
        self.ydata = ydata

    def addNewLevel(self,ErrorThreshold: float,maxLvl: int):
        '''
        Add New Level to Hierarchical Basis:
        ====================================

        Recursive implementation for adding new hierarchical level of basis functions. 
        Each node first determines its scaling coefficient and calculates the respective surplus. 
        The data and the surpluses is split into the two hierarchical lower intervals left and right of the midpoint. 
        If the maximal surplus of the split data is smaller than the ErrorThreshold we stop.
        Otherwise a new hierarchical basis function of half the domain is introduced by a recursive call. 
        The other termination criterion is if there exists no point to approximate in the respective split interval. 
        The structure storing the hierarchical basis can therefore be interpreted as a tree. 

        Parameters:
        -----------
        ErrorThreshold: float
            Error threshold for termination of hierarchical levels
        maxLvl: int
            Number of maximal layers for termination criterion.
        '''
        h = 1.0/(2**self.level)

        # Old scaling scheme??
        #yscale = [y*hnmh.hatFunction(self.xpoint,h,x) for x,y in zip(self.xdata,self.ydata)]
        #self.coeff = sum(yscale)/len(yscale)

        self.coeff = sum(self.ydata)/len(self.ydata)

        self.ydata = [y-self.coeff*hnmh.hatFunction(self.xpoint,h,x) for x,y in zip(self.xdata,self.ydata)]
        
        midpointleft = self.xpoint - h/2
        midpointright = self.xpoint + h/2
        inright = [x >= self.xpoint and self.xpoint < self.xpoint+h for x in self.xdata]
        inleft = [x < self.xpoint and self.xpoint > self.xpoint-h for x in self.xdata]
        xright = []
        xleft = []
        yright = []
        yleft = []

        for i in range(0,len(self.xdata)):
            if inright[i] == 1:
                xright.append(self.xdata[i])
                yright.append(self.ydata[i])
            if inleft[i] == 1:
                xleft.append(self.xdata[i])
                yleft.append(self.ydata[i])
        if len(yleft) > 0:
            if (max(yleft) > ErrorThreshold or min(yleft) < -ErrorThreshold)and self.level < maxLvl:
                self.left = Node(midpointleft,self.level+1,xleft,yleft)
                self.left.addNewLevel(ErrorThreshold,maxLvl)
        if len(yright) > 0:
            if (max(yright) > ErrorThreshold or min(yright) < -ErrorThreshold)and self.level < maxLvl:
                self.right = Node(midpointright,self.level+1,xright,yright)
                self.right.addNewLevel(ErrorThreshold,maxLvl)

    def evaluateNodes(self,x: float) -> float:
        '''
        Evaluation of Hierarchical basis:
        =================================

        This functions evaluates the hierarchical basis for a given point x. It is a recursive function call traversing
        through the tree structure, which stores the hierarchical basis. At each node only the interval containing x is evaluated, such that we exploit
        the structure of a binary search tree. 

        Parameters:
        -----------
        x: float
            The x parameter is a given float

        Returns:
        --------
        y: float
            The evaluated value at position x
        '''
        h = 1.0/(2**self.level)
        xhat = hnmh.hatFunction(self.xpoint,h,x)
        yscale = self.coeff*xhat
        if x < self.xpoint:
            if self.left != 0:
                return yscale + self.left.evaluateNodes(x)
            else:
                return yscale
        if x > self.xpoint:
            if self.right != 0:
                return yscale + self.right.evaluateNodes(x)
            else:
                return yscale
       
        return yscale
        
    def plotBasis(self):
        '''
        Plot Hierarchical Basis:
        ========================

        This function plots all basis functions contained in the tree recursively, such that a visual intuition of the basis functions can be gained. 
        '''
        h = 1.0/(2**self.level)
        plt.plot([self.xpoint-h,self.xpoint,self.xpoint+h],[0,self.coeff,0])
        if self.left != 0:
            self.left.plotBasis()
        if self.right != 0:
            self.right.plotBasis()

'''
Callable Functions:

These functions are available from outside the module.
'''

def plinint(xvec: list, yvec: list, x):
    """
    Piecewise linear interpolation
    ==============================

    This function returns a piecewise linear interpolation of a point x or list of points x, for a given vector of x and y values. 

    Parameters:
    -----------
    xvec: list
        list of given x values. x values are sorted, such that also unsorted x points can be inputs.
    yvec: list
        list of given y values for xvec points
    x: list or float: 
        x value(s) at which the function shall be interpolated.

    Returns:
    --------
    y: list or float:
        y value(s) for the interpolation point(s) x

    Raises: 
    -------
    Value Error:
        If the input sizes of xvec and yvec are not sufficient.
    Value Error: 
        If the interpolation point(s) x are given out of the interval xvec.
    """
    if len(xvec) != len(yvec) or len(xvec) < 2:
        raise ValueError('Input size of xvec and yvec have to be equal and larger than 1')
    
    xv,yv = hnmh.order(xvec,yvec)
    
    if type(x) != list:
        if x < xv[0] or x > xv[1]:
            raise ValueError('Interpolation point(s) are out of bounds.')
        
        return sum([yv[i] * hnmh.phihat(x,i,xv) for i in range(0,len(xv))])
    else:
        if x[0] < xv[0] or x[-1] > xv[-1]:
            raise ValueError('Interpolation point(s) are out of bounds.')
        
        return [sum([yv[i] * hnmh.phihat(x[j],i,xv) for i in range(0,len(xv))]) for j in range(0,len(x))]


class Classifier1Dnodal:
    '''
    1D Classifier based on nodal nasis functions
    ============================================

    Classifier is traned on (x,y) data and can then predict y for unseen x by interpoating with the nodal basis.
    All input data x is normalised onto the interval [0,1]. 
    Categorical analysis is not included and can be done in an additional postprocessing step.

    Initialisation:
    ---------------
    n int: 
        During initialisation please give the number of basis functions that shall be used. 

    Attributes:
    -----------
    trained: bool
        Boolean for recogbising state of classifier. True: training dataset was applied.
    n int:
        Number of nodal basis fnctions used for interpolation.
    v: list
        Result vector of basis function coefficients from training, later used in classification.
    minv: float
        Minimal value of input data used for normlaisiation. Same normalisation scheme also for classification.
    maxv: float
        Maxial value of input data used for normalisation. Same normalisation scheme also used for classification. 
    
    Functions:
    ----------
    train: 
        Train the classifier based on example data. 
    classify:
        Classify unseen data.
    '''
    def __init__(self, n: int):
        self.trained = False
        self.n = n
        self.v = []
        self.minv = 0
        self.maxv = 1

    def train(self, xvec: list, yvec: list):
        '''
        Train 1D Classifier Nodal Basis
        ===============================

        This function trains the classifier on known data (x,y) 
        and determins the coefficients for the nodal basisi functions. 

        Parameters:
        -----------
        xvec: list
            vector of xvalues (data). The vector does not have to be sorted or normalised.
        yvec: list
            vector of yvalues (classes). The vector is sorted according to the xvector.
        
        Returns:
        --------
        None but the coefficients are internally stored in vector 'v'.
        '''
        xv,yv = hnmh.order(xvec,yvec)
        self.minv = min(xv)
        self.maxv = max(yv)
        xn = hnmh.normalize(xv,self.minv,self.maxv)

        a,b,c,d = hnmh.setuplgs(xn,yvec,self.n)

        self.v = hnmh.TDMlsgsolver(a,b,c,d)
        self.trained = True

    def classify(self,x):
        '''
        Classify 1D Classifier Nodal Basis
        ==================================

        This function classifys unseen data x. 

        Parameters: 
        -----------
        x: list or float
            The given input can be a single float or a list of floats, which are normalised in the 
            equivalent way as the data of the training dataset. 

        Return:
        -------
        out: list or float
            The given output is the result of the interpolation, which can be used for classifaction. 

        Raises: 
        -------
        ValueError:
            If the input is not in the given maximal interval of the training dataset seen before.
        ValueError:
            If the classifier is not trained yet.
        '''
        if self.trained:
            h = 1/(self.n+1)
            xi = [i*h for i in range(0,self.n+2)]

            if type(x) == list:
                if max(x) > self.maxv or min(x) < self.minv:
                    raise ValueError('Given interpolation data not in the required interval used during training.')
                xn = hnmh.normalize(x,self.minv,self.maxv)
                return [sum([self.v[j]*hnmh.phinodal(x[i],self.n,j+1,xi) for j in range(0,self.n)]) for i in range (0,len(x))]
            else:
                if x > self.maxv or x < self.minv:
                    raise ValueError('Given interpolation data not in the required interval used during training.')
                xn = (x-self.minv)/self.maxv
                return sum([self.v[j]*hnmh.phinodal(x,self.n,j+1,xi) for j in range(0,self.n)])
        else:
            raise ValueError('Classifier not trained yet. Please train the classifier first by calling .train(...)')


def quadrature1Dcomp(f,a: float,b: float,n: int, vers: str = 'trap') -> float:
    '''
    Numerical Quadrature 1D Composition Schemes
    ===========================================

    This function implements the compositional trapezoidal or simpson scheme.

    Parameters:
    -----------
    f: function
        Input is a function f which is defined as f(float) -> float.
    a: float
        Integration boundary lower bound.
    b: float
        Integration boundary upper bound.
    n: int
        Number of used intervals.
    vers: str
        String identifier for the computation scheme. 

    Returns:
    --------
    integral: float
        Numerically evaluated integral of f in integration bounds [a,b]

    Raises:
    -------
    ValueError:
        Unknown string identifier for vers. 
    '''
    integral = 0.0
    dh = (b-a)/float(n)

    if vers == 'trap':
        for i in range(n):
            l = a+i*dh
            r = l+dh
            integral += dh*(f(l)+f(r))/2
    elif vers == 'simp':
        for i in range(n):
            l = a+i*dh
            r = l+dh
            integral += dh*(f(l)+4*f((l+r)/2)+f(r))/6
    else:
        raise ValueError('Unknown string identifier {str} for vers. Vers is required to be (1) trap or (2) simp.')

    return integral

def hierarchise1D(u: list) -> list:
    '''
    Hierarchical Transformation 
    ===========================

    This functions performs the transform from the nodal basis to hierarchical basis. 
    Given a set of function values at evenly spaced points. Assuming boundray conditions u_0 = 0, u_n = 0.

    Parameters:
    -----------
    u: list
        list of evenly spaced function values (nodal basis). Has to have size 2**l-1

    Returns:
    --------
    v: list
        coefficients of hierarchical basis

    Raises:
    -------
    ValueError:
        If the input size of u is not as required 2**l-1
    '''
    N = len(u)
    if math.log2(N+1) % 1 != 0 or N < 1:
        raise ValueError(
        "Invalid input list size. Input list size is expected to have len 2**n-1 and > 0"
        )
    maxl = int(math.log2(N+1))
    v = list(u)

    for l in range(maxl-1, 0, -1):
        
        delta_next = 1 << (maxl-l-1)    
        first_this = (1<< (maxl-l)) - 1  
        
        for j in range(first_this, N, delta_next<<1):
            v[j-delta_next] -= 0.5 * u[j]
            v[j+delta_next] -= 0.5 * u[j]
        
    return v

def dehierarchise1D(v: list) -> list:
    '''
    Inverse Hierarchical Transformation 
    ===========================

    This functions performs the transform from the nodal basis to hierarchical basis. 
    Given a set of function values at evenly spaced points. Assuming boundray conditions u_0 = 0, u_n = 0.

    Parameters:
    -----------
    v: list
        list of hierarchical basis coefficients. Has to have size 2**l-1

    Returns:
    --------
    u: list
        nodal basis coefficents (evenly spaced grid points)

    Raises:
    -------
    ValueError:
        If the input size of v is not as required 2**l-1
    '''
    N = len(v)
    if math.log2(N+1) % 1 != 0 or N < 1:
        raise ValueError(
        "Invalid input list size. Input list size is expected to have len 2**n-1 and > 0"
        )
    maxlv = int(math.log2(N+1))
    u = list(v)

    for l in range(1, int(maxlv+1)):
        delta = 2**(maxlv - l)
        start = delta - 1
        for i in range(0, 2**l - 1, 2):
            position = start + i * delta
            assert(N > position >= 0)
            u[position] = v[position]
            if position - delta >= 0:
                u[position] += 0.5 * u[position - delta]
            if position + delta < N:
                u[position] += 0.5 * u[position + delta]
        
    return u

class Classifier1DSparse:
    '''
    1D Classifier based on sparse hierarchical basis functions
    ==========================================================

    Classifier is traned on (x,y) data and can then predict y for unseen x by interpoating with the sparse hierarchical basis functions.
    All input data x is is required to be in the interval [0,1] with boundary conditions u_0 = e_end = 0. 
    Categorical analysis is not included and can be done in an additional postprocessing step.

    Initialisation:
    ---------------
    The classifier is initialised generically and all spcifications are made in the training call.

    Attributes:
    -----------
    trained: bool
        Boolean for recogbising state of classifier. True: training dataset was applied.
    n int:
        Number of nodal basis fnctions used for interpolation.
    root: Node
        Node of the result tree from training.
    
    Functions:
    ----------
    train: 
        Train the classifier based on example data. 
    classify:
        Classify unseen data.
    '''
    def __init__(self):
        self.trained = False
        self.maxv = 1
        self.minv = 0
    
    def train(self,xdata: list ,ydata: list ,ErrorThreshold: float ,maxLvl : int):
        '''
        Train 1D Sparse Classifier
        ==========================

        This function trains the 1D Sparse Classifier based on hierarchical basis functions. 
        A sparse hierarchical basis is developed in a recursive scheme. 

        Parameters:
        -----------
        xdata: list
            List of xdata. xdata has to be in the interval [0,1] (normalised)
        ydata: list
            List of ydata. ydata has to fulfill y(x=0)=0 and y(x=1)=0. 
        ErrorThreshold: float
            Absolute error threshold, which shall be sattisfied.
        mxLvl: float
            A maximal hierarchical basis level maxLvl>0.
        
        Returns:
        --------
        None since the tree structure with the hierarchical coefficients is stored internally in self.root (root node of the tree).

        Raises:
        -------
        ValueError: 
            If xdata is not normalised to interval [0,1]
        ValueError:
            If ydata doesn't satisfy the boundary conditions y(x=0)=0 and y(x=1)=0. 
        ValueError:
            If xdata and ydata have differnet number of elements or are empty
        ValueError:
            If maxLvl is no positive integer > 0.
        '''

        xdata,ydata = hnmh.order(xdata,ydata)

        if len(xdata) == 0 or len(xdata) != len(ydata):
            raise ValueError('Lists xdata and ydata have to be none empty lists of equal length.')
        if max(xdata)>1 or min(xdata)<0:
            raise ValueError('The xdata list has to be normalised in the interval [0,1]')
        if (xdata[0] == 0 and ydata[0] != 0) or (xdata[-1] == 1 and ydata[-1] != 0):
            raise ValueError('The boundray condition of the ydata y(x=0)=0 and y(x=1)=0 has to be satisfied.')
        if type(maxLvl) != int or maxLvl <= 0:
            raise ValueError('maxLvl has to be a positive intheger > 0')

        self.root = Node(0.5,1,xdata,ydata)
        self.root.addNewLevel(ErrorThreshold,maxLvl)
        self.trained = True

    def classify(self,x)->float:
        '''
        Classify 1D Sparse Classifier
        =============================

        This function returns the interpolated value (label) for a given input, which can then be used for subsequent classification. 

        Parameters:
        -----------
        x: list or float 
            xdata for classification. Has to be normalised in the inetrval [0,1]. Input data can either be a float or a list of floats

        Returns:
        --------
        y: list or float
            Interpolated value. Has the type of the x input data.

        Raises:
        -------
        ValueError:
            If x is empty or not normalised in the interval [0,1].
        ValueError:
            If Classifier is not trained yet.
        '''
        if self.trained:
            if type(x) == list:
                if len(x) == 0:
                    raise ValueError('List od xdata has to be a non empty list')
                if max(x) > 1 or min(x)<0:
                    raise ValueError('xdata has to be nomrlaised in [0,1]')
                return [self.root.evaluateNodes(xi) for xi in x]
            else:
                if x > 1 or x < 0:
                    raise ValueError('xdata has to be nomrlaised in [0,1]')
                return self.root.evaluateNodes(x)
        else:
            raise ValueError('Classififer not trained yet.')
        
    def plotBasis(self, show: bool):
        '''
        Plotting Sparse Hierarchical Basis:
        ===================================

        The sparse Hierarchical basis functios are plotted.

        Parameters:
        -----------
        show: bool
            Decides if plot is shown emmidately or if additional plots can be added to the figure.
        '''
        plt.figure()
        self.root.plotBasis()
        if show:
            plt.show()


    

    

