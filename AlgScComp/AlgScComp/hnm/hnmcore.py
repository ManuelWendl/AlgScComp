"""
Hierarchical Numerical Methods
==============================

This module contains the following algporithms: 

plinint(xvec, yvec, x) - piecewise linear interpolation

"""

from . import hnmhelper as hnmh

__all__ = ["plinint","Classifier1Dnodal"]

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
                if max(x) > self-maxv or min(x) < self.minv:
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










    

    

