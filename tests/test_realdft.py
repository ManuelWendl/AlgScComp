import AlgScComp as asc
import numpy as np

f = [1,2,3,4,5,6,7,8]

f2 = [f,f]


print(asc.dft.rfft(f,'inv'))
print(asc.dft.fft(f,'inv'))
