import AlgScComp as asc
import numpy as np
import time 


a = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
b = [8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0]

f = [a[i]+1J*b[i] for i in range(0,len(a))]

print('FFTshift:')
print(asc.dft.fftshift(f))
print('---------')


print('System FFT')
print(np.fft.fft(f)*1/len(f))
print('---------')
print(np.fft.ifft(np.fft.fft(f)))
print('---------')


print('DFT:')
print('---------')
print(asc.dft.dft(f))
print('---------')
print(asc.dft.idft(asc.dft.dft(f)))
print('---------\n')


print('FFT recursive')
print('---------')
print(asc.dft.fft(f,'fwd','rec'))
print('---------')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','rec'),'fwd','rec'))
print('---------\n')


print('FFT v1')
print('---------')
print(asc.dft.fft(f,'fwd','v1','c'))
print(asc.dft.fft(f,'fwd','v1','py'))
print('---------')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','v1'),'fwd','v1'))
print('---------\n')


print('FFT vectorised')
print('---------')
print(asc.dft.fft(f,'fwd','vec','c'))
print(asc.dft.fft(f,'fwd','vec','py'))
print('---------\n')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','vec'),'fwd','vec'))
print('---------\n')


print('FFT v2')
print('---------')
print(asc.dft.fft(f,'fwd','v2'))
print('---------\n')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','v2'),'fwd','v2'))