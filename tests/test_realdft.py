import AlgScComp as asc
import scipy
import numpy as np

f = [0,-1,2,4,5]

f2 = [f,f]


#print(asc.dft.rfft(f,'inv'))
#print(asc.dft.fft(f,'inv'))
#print(scipy.fftpack.fft(f))

import math
def dct(g):
    N = len(g)-1
    return [1/N*(sum([g[n]*math.cos(math.pi*k*n/N) for n in range(1,N)])+g[0]/2+g[N]/2*math.cos(math.pi*k)) for k in range(N+1)]

def dctinv(g):
    N = len(g)-1
    return [(sum([g[n]*math.cos(math.pi*k*n/N) for n in range(1,N)])+g[0]/2+g[N]/2*math.cos(math.pi*k)) for k in range(N+1)]

#print('\n\n')
#print(asc.dft.fct(f,'fwd'))
#print(dct(f))
#print(dctinv(f))

#print('\n\n')
#print(asc.dft.ifct(asc.dft.fct(f,'fwd'),'fwd'))

def qwdct(g):
    N = len(g)
    return 1/N * np.array([np.sum([f[n]*np.cos((np.pi*k*(n+0.5))/N) for n in range(0,N)]) for k in range(0,N)])

def inverse_qwdct(F):
    N = len(F)
    
    if math.log(N, 2) % 1 != 0 or N < 1:
        raise ValueError("Invalid input list size. Input list size is expected to have len 2**n and > 0")
    
    G = F + list(reversed(F))
    g = np.fft.ifft(G).real
    
    result = []
    for i in range(N):
        angle = (2 * i + 1) * math.pi / (4 * N)
        cos_term = math.cos(angle)
        sin_term = math.sin(angle)
        result.append(g[i] * (cos_term + 1j * sin_term))
    
    return result

f = [1,2,3,4,5,6,7,8]
#print(qwdct(f))
#print(asc.dft.qwfct(f))
#print(asc.dft.iqwfct(asc.dft.qwfct(f,'inv'),'inv'))

f = [0,1,2,3,4,5,6,7]
N = len(f)
#print(np.array(-1J/N * np.array([np.sum([f[n]*np.sin((np.pi*k*n)/N) for n in range(1,N)]) for k in range(1,N)])))
f = [1,2,3,4,5,6,7]
print('DST: ',asc.dft.fst(f,'inv'))
print(asc.dft.ifst(asc.dft.fst(f,'inv'),'inv'))
