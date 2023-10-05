import AlgScComp as asc
import numpy as np
import time 
import matplotlib.pyplot as plt

testRuntime = True

if testRuntime == False:
    a = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
    b = [8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0]
else:
    a = np.random.randn(1,2**8)*10
    b = np.random.randn(1,2**8)*10
    a = a.tolist()
    b = b.tolist()
    a = a[0][:]
    b = b[0][:]

f = [a[i]+1J*b[i] for i in range(0,len(a))]

print('FFTshift:')
if testRuntime == False:
    print(asc.dft.fftshift(f))
print('---------')


print('System FFT')
if testRuntime == False:
    print(np.fft.fft(f)*1/len(f))
print('---------')
if testRuntime == False:
    print(np.fft.ifft(np.fft.fft(f)))
print('---------')


print('DFT:')
print('---------')
if testRuntime == False:
    print(asc.dft.dft(f))
print('---------')
if testRuntime == False:
    print(asc.dft.idft(asc.dft.dft(f)))
print('---------\n')


print('FFT recursive')
print('---------')
print('C Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','rec','c')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('Python Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','rec','py')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('---------')
if testRuntime == False:
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','rec','c'),'fwd','rec','c'))
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','rec','py'),'fwd','rec','py'))
print('---------\n')

if testRuntime == False:
    print('\n\n',f,'\n\n')

print('FFT v1')
print('---------')
print('C Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','v1','c')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('Python Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','v1','py')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('---------')
if testRuntime == False:
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','v1','c'),'fwd','v1','c'))
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','v1','py'),'fwd','v1','py'))
print('---------\n')

if testRuntime == False:
    print('\n\n',f,'\n\n')

print('FFT vectorised')
print('---------')
print('C Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','vec','c')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('Python Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','vec','py')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('---------\n')
if testRuntime == False:
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','vec','c'),'fwd','vec','c'))
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','vec','py'),'fwd','vec','py'))
print('---------\n')

if testRuntime == False:
    print('\n\n',f,'\n\n')

print('FFT v2')
print('---------')
print('C Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','v2','c')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('Python Implementation:')
tstart = time.time()
F = asc.dft.fft(f,'fwd','v2','py')
print('Elapsed Time: ',time.time()-tstart)
if testRuntime == False:
    print(F)
print('---------\n')
if testRuntime == False:
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','v2','c'),'fwd','v2','c'))
    print(asc.dft.ifft(asc.dft.fft(f,'fwd','v2','py'),'fwd','v2','py'))

c = [[1,2,3,4,5,6],[1,2,3,4,5,6]]
d = [1,2,3,4,5,6]

W = asc.dft.dftMat(265)

plt.figure()
plt.imshow(np.real(np.array(W)),cmap='gray')
plt.show()

x = np.linspace(10,20,128)
f = np.sin(2*np.pi*x) + np.sin(2*5*np.pi*x)

fl = f.tolist()

P,Freq = asc.dft.powerSpectrum(fl,10)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x,f)
plt.subplot(2,1,2)
plt.plot(np.array(Freq),np.array(P))
plt.show()