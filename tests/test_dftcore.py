from AlgScComp import dft

f = [1, 2, 3, 4, 5, 6, 7, 8]
print(dft.fft_rek(f))
print('---------')
print(f)
print('---------')
print(dft.dft(f))
print('---------')
print(f)
print('---------')
print(dft.fft_itt_v1(f))
print('---------')
print(f)