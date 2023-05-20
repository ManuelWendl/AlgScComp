import AlgScComp as asc

a = [1,2,3,4,5,6,7,8] 
b = [8,7,6,5,4,3,2,1]

f = [a[i]+1J*b[i] for i in range(0,len(a))]

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
print(asc.dft.fft(f,'fwd','v1'))
print('---------')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','v1'),'fwd','v1'))
print('---------\n')
print('FFT vectorised')
print('---------')
print(asc.dft.fft(f,'fwd','vec'))
print('---------\n')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','vec'),'fwd','vec'))
print('---------\n')
print('FFT v2')
print('---------')
print(asc.dft.fft(f,'fwd','v2'))
print('---------\n')
print(asc.dft.ifft(asc.dft.fft(f,'fwd','v2'),'fwd','v2'))