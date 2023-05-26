import AlgScComp as asc
import numpy as np
import time
import matplotlib.pyplot as plt

times_v1 = np.zeros((8,2))
times_vec = np.zeros((8,2))
times_v2 = np.zeros((8,2))

j = 0

for i in range(5,21,2):

    f = list(np.random.randn(2**i) + 1J*np.random.randn(2**i))

    tc = time.time()
    Fc = asc.dft.fft(f,vers='v1',lang='c')
    elapsedc = time.time() - tc
    tp = time.time()
    Fc = asc.dft.fft(f,vers='v1',lang='py')
    elapsedp = time.time() - tp

    times_v1[j,:] = [elapsedc,elapsedp]

    tc = time.time()
    Fc = asc.dft.fft(f,vers='vec',lang='c')
    elapsedc = time.time() - tc
    tp = time.time()
    Fc = asc.dft.fft(f,vers='vec',lang='py')
    elapsedp = time.time() - tp

    times_vec[j,:] = [elapsedc,elapsedp]

    tc = time.time()
    Fc = asc.dft.fft(f,vers='v2',lang='c')
    elapsedc = time.time() - tc
    tp = time.time()
    Fc = asc.dft.fft(f,vers='v2',lang='py')
    elapsedp = time.time() - tp

    times_v2[j,:] = [elapsedc,elapsedp]

    j += 1

plt.figure()
plt.subplot(1,3,1)
plt.semilogy(times_v1[:,0])
plt.semilogy(times_v1[:,1])
plt.semilogy(times_vec[:,0])
plt.semilogy(times_vec[:,1])
plt.semilogy(times_v2[:,0])
plt.semilogy(times_v2[:,1])
plt.legend(['v1:c','v1:py','vec:c','vec:py','v2:c','v2:py'])
plt.title('All combinations')
plt.ylabel('log(t)')
plt.xlabel('p (N=2^p)')

plt.subplot(1,3,2)
plt.semilogy(times_v1[:,0])
plt.semilogy(times_vec[:,0])
plt.semilogy(times_v2[:,0])
plt.legend(['v1:c','vec:c','v2:c'])
plt.title('C only')
plt.ylabel('log(t)')
plt.xlabel('p (N=2^p)')

plt.subplot(1,3,3)
plt.semilogy(times_v1[:,1])
plt.semilogy(times_vec[:,1])
plt.semilogy(times_v2[:,1])
plt.legend(['v1:py','vec:py','v2:py'])
plt.title('Py only')
plt.ylabel('log(t)')
plt.xlabel('p (N=2^p)')
plt.show()

print(times_v1)
print(times_vec)
print(times_v2)