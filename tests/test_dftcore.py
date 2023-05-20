from AlgScComp import dft

a = [1,2,3,4,5,6,7,8] 
b = [8,7,6,5,4,3,2,1]

f = [a[i]+1J*b[i] for i in range(0,len(a))]

print('DFT:')
print('---------')
print(dft.dft(f))
print('---------')
print(f)
print('---------\n')
print('FFT_rek')
print('---------')
print(dft.fft_rek(f))
print('---------')
print(f)
print('---------\n')
print('FFT_itt_v1')
print('---------')
print(dft.fft_itt_v1(f))
print('---------')
print(f)
print('---------\n')
print('FFT_itt_vec')
print('---------')
print(dft.fft_itt_vec(f))
print('---------\n')
print(f)