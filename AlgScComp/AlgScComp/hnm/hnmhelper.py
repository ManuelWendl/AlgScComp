"""
Helper Functions for the Hierarchical Numerical Methods 

"""

def phihat(x: float, i: int, xvec: list) -> float: 
    """
    Hat basis function for interval x_i-1 < x < x_i+1 with peak at x_i.
    """
    if i>0 and i<len(xvec):
        if x <= xvec[i-1] or x>= xvec[i+1]:
            return 0
        elif x > xvec[i-1] and x <= xvec[i]:
            h = xvec[i]-xvec[i-1]
            return 1/h*(x-xvec[i-1])
        else:
            h = xvec[i+1]-xvec[i]
            return 1/h*(xvec[i+1]-x)
    elif i==0:
        if x>= xvec[i+1]:
            return 0
        else: 
            h = xvec[i+1]-xvec[i]
            return 1/h*(xvec[i+1]-x)
    else:
        if x <= xvec[i-1]:
            return 0
        else:
            h = xvec[i]-xvec[i-1]
            return 1/h*(x-xvec[i-1])

def phinodal(x: float,n: int, i: int, xvec: list) -> float:
    """
    Equally spaced hat basis function for interval x_i-1 < x < x_i+1 with peak at x_i.
    Also called nodal basis functions.
    """
    h = 1/(n+1)

    if i>0 and i<len(xvec):
        if x <= xvec[i-1] or x>= xvec[i+1]:
            return 0
        elif x > xvec[i-1] and x <= xvec[i]:
            return 1/h*(x-xvec[i-1])
        else:
            return 1/h*(xvec[i+1]-x)
    elif i==0:
        if x>= xvec[i+1]:
            return 0
        else: 
            return 1/h*(xvec[i+1]-x)
    else:
        if x <= xvec[i-1]:
            return 0
        else:
            h = xvec[i]-xvec[i-1]
            return 1/h*(x-xvec[i-1])
        
def order(xvec:list, yvec:list):
    '''
    Sorting input points for ascending x values. 
    '''
    x = xvec.copy()
    y = yvec.copy() 
    xy = sorted(zip(x, y))
    x,y = zip(*xy)
    return list(x),list(y)
    
def normalize(vec, minv: float, maxv: float):
    '''
    Normalising the input in [0.1,0.9]
    The reason for choosing the interval is that our basis functions can only consider boundary conditions u_0 = u_end = 0
    '''
    if type(vec) == list:
        vec = [x-minv for x in vec]
        vec = [x/maxv*0.8+0.1 for x in vec]
    else:
        vec = (vec-minv)/maxv*0.8+0.1
    return vec
    
def setuplgs(xvec: list, yvec: list, n: int):
    '''
    Setup the linear system of equations in a sparse tridiagonal matrix representation with upper diagonal a, middle b and lower c
    The matrix product G^T*G and vector product G^T*yvec are computed in an efficient way by considering the tridiagonal structure. 
    '''
    h = 1/(n+1)
    xi = [i*h for i in range(0,n+2)]
    G = [[phinodal(xvec[i],n,j+1,xi) for i in range(0,len(xvec))] for j in range(0,n)]

    a = []  # Tridiagonal matrix (upper diagonal)
    b = []  # Tridiagonal matrix (middle diagonal)
    c = []  # Tridiagonal matrix (lower diagonal)
    d = []  # solution vector G * yvec

    for i in range(0, len(G)):
        di = sum(G[i][j]*yvec[j] for j in range(0,len(xvec)))
        d.append(di)
        bi = sum([Gi**2 for Gi in G[i]])
        b.append(bi)
        if i < len(G)-1:
            ai = sum([G[i][j] * G[i+1][j] for j in range(0,len(xvec))])
            a.append(ai)
        if i > 0:
            ci = sum([G[i][j] * G[i-1][j] for j in range(0,len(xvec))])
            c.append(ci)

    return a,b,c,d

def TDMlsgsolver(a: list,b: list,c: list,d: list):
    '''
    Solver for a Tridiagonal matrix system of equations. 
    a upper diagunal, b middle diagonal and c lower diagonal. 
    d is the solution vector and x the desired coefficients. 
    '''
    n = len(d)

    for i in range(1,n):
        m = a[i-1]/b[i-1]
        b[i] = b[i] - m*c[i-1]
        d[i] = d[i] - m*d[i-1]
    
    x = b
    x[-1] = d[-1]/b[-1]

    for i in range(n-2,-1,-1):
        x[i] = (d[i]-c[i]*x[i+1])/b[i]
    
    return x

def hatFunction(midpoint,h,x):
    '''
    Helper function for the hierarchical classifier.
    Returning the hat function based on an interval. The most used cases are taken as the two if statements to optimise runtime. 
    Case 3 should not be accessed during the hierarchical sparse classification. 
    '''
    if (x >= midpoint):
        return (midpoint+h - x)/h
    elif (x < midpoint):
        return (x-(midpoint-h))/h
    else:
        return 0

def waveletEdgeTreat(c,edgeTreat,overhead):
    '''
    Helper function for the wavelet transform:
    For Daubechies wavelets grater than Haar we require boundray treatments.
    Can be periodic, mirrored or filled with zeros.
    '''
    if overhead > 0:
        if edgeTreat == 'periodic':
            c.extend(c[0:overhead])
        elif edgeTreat == 'mirror':
            c.extend(c[-1:-overhead-1:-1])
        elif edgeTreat == 'zeros':
            c.extend([0]*overhead)
        else:
            raise ValueError('String identifier {edgeTreat} for edgeTreat not recognised. EdgeTreat can be selected from (1) periodic, (2) mirrored, (3) zeros')
    return c

def iwaveletEdgeTreat(c,edgeTreat,overhead):
    '''
    Helper function for the wavelet transform:
    For Daubechies wavelets grater than Haar we require boundray treatments.
    Can be periodic, mirrored or filled with zeros.
    '''
    if overhead > 0:
        if edgeTreat == 'periodic':
            c = c[-int(overhead/2):]+c
        elif edgeTreat == 'mirror':
            c = c[-1:-int(overhead/2)-1:-1]+c
        elif edgeTreat == 'zeros':
            c = [0]*int(overhead/2) + c
        else:
            raise ValueError('String identifier {edgeTreat} for edgeTreat not recognised. EdgeTreat can be selected from (1) periodic, (2) mirrored, (3) zeros')
    return c
        

def waveletstep(c,p,q,edgeTreat,overhead):
    '''
    One wavelet transformation step implementation.
    General Daubechies implementation with Daubechies coefficients p,q
    '''
    cCopy = waveletEdgeTreat(c.copy(),edgeTreat,overhead)
    cc = [sum([p[i-2*j] *cCopy[i] for i in range(2*j,2*j+len(p))]) for j in range(0, int(len(c)/2))]
    dc = [sum([q[i-2*j] *cCopy[i] for i in range(2*j,2*j+len(q))]) for j in range(0, int(len(c)/2))]
    return cc + dc

def iwaveletstep(c,p,q,edgeTreat,overhead):
    '''
    ! Be careful 
        Only for Haar Wavelets correct!!
    !
    One inverse wavelet transformation step implementation.
    General Daubechies implementation with Daubechies coefficients p,q
    '''
    ctreated = iwaveletEdgeTreat(c[:int(len(c)/2)].copy(),edgeTreat,overhead)
    dtreated = iwaveletEdgeTreat(c[int(len(c)/2):].copy(),edgeTreat,overhead)
    cc = []

    for i in range(0,int(len(c)),2): 
        cc.append(sum([p[j]*ctreated[int(i/2)+int((len(q)-2)/2)-int(j/2)]+q[j]*dtreated[int(i/2)+int((len(q)-2)/2)-int(j/2)] for j in range(len(q)-2,-1,-2)]))
        cc.append(sum([p[j+1]*ctreated[int(i/2)+int((len(q)-2)/2)-int(j/2)]+q[j+1]*dtreated[int(i/2)+int((len(q)-2)/2)-int(j/2)] for j in range(len(q)-2,-1,-2)]))
    return cc

def checkDimensions(x: list) -> list:
    '''
    Checks correct dimensions of matrix u for 2D hierarchisation
    '''
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