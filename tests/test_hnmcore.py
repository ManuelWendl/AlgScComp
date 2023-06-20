import AlgScComp as asc
import random
import matplotlib.pyplot as plt
import math

xvec = [0,1,2,3,4,5,6,7,8,9]
yvec = [random.random() for i in range(0,10)]

xint = [0, 1, 2.1, 3.8, 4.3, 5.1]
yint = asc.hnm.plinint(xvec,yvec,xint)

#plt.figure()
#plt.plot(xvec,yvec)
#plt.scatter(xint,yint)
#plt.show()


S = [(0,0),(0.2,1),(0.3,-1),(0.35,1),(0.4,1),(0.55,-1),(0.6,-1),(0.65,-1),(0.7,-1),(1,0)]
#vector with 1D features
xVec = [s[0] for s in S]
#vector with labels of the training data
yVec = [s[1] for s in S]

C = asc.hnm.Classifier1Dnodal(3)
C.train(xVec,yVec)
print(C.classify([0.5,0.7]))

def f(x):
    return math.sin(x)

print(asc.hnm.quadrature1Dcomp(f,0,math.pi,1000,'trap'))
print(asc.hnm.quadrature1Dcomp(f,0,math.pi,1000,'simp'))

u = [0.4375,0.7500,0.9375,1.000,0.9375,0.7500,0.4375]
print(asc.hnm.hierarchise1D(u))
print(asc.hnm.dehierarchise1D(asc.hnm.hierarchise1D(u)))


C = asc.hnm.Classifier1DSparse()
C.train(xVec,yVec,0.1,10)
y = C.classify(0.5)
print(y)

#C.plotBasis(False)

#plt.plot(xVec,yVec,'-ok')
#y = [C.classify(i/100) for i in range(0,100)]
#x = [i/100 for i in range(0,100)]
#plt.plot(x,y,'r')
#plt.show()


p1 = [1/math.sqrt(2),1/math.sqrt(2)]
q1 = [1/math.sqrt(2),-1/math.sqrt(2)]

p2 = [(1+math.sqrt(3))/(4*math.sqrt(2)),(3+math.sqrt(3))/(4*math.sqrt(2)),(3-math.sqrt(3))/(4*math.sqrt(2)),(1-math.sqrt(3))/(4*math.sqrt(2))]
q2 =  [(1-math.sqrt(3))/(4*math.sqrt(2)),-(3-math.sqrt(3))/(4*math.sqrt(2)),(3+math.sqrt(3))/(4*math.sqrt(2)),-(1+math.sqrt(3))/(4*math.sqrt(2))]

print(p2,q2)

s2 = [8, 4, -1, 1, 0, 4, 1, 7, -5/2, -3/2, 0, -4, -2, -2, 1, -5]
print(asc.hnm.wavelet1D(s2,p2,q2,edgeTreat='periodic'))
print('Inverse Test')
a = asc.hnm.wavelet1D(s2,p2,q2,edgeTreat='zeros',minLvl=1)
b= asc.hnm.iwavelet1D(a,p2,q2,edgeTreat='zeros',minLvl=1)
print('b=',b)
print('Inverse Test')
a = asc.hnm.wavelet1D(s2,p1,q1,edgeTreat='mirror')
b= asc.hnm.iwavelet1D(a,p1,q1,edgeTreat='mirror')
print('b=',b)

a = asc.hnm.wavelet2D([[4,2,3,5],[1,-7,0,8],[-1,-3,9,-3],[6,-2,-1,1]],p1,q1)
print(a)
ainv =  asc.hnm.iwavelet2D(a,p1,q1)
print(ainv)

import os
import numpy as np

path = os.path.dirname(__file__)
from skimage.io import imread
from skimage.color import rgb2gray
im = imread(path+"/testimagewavelet.jpg")
im = rgb2gray(im).tolist()

p1 = [1/math.sqrt(2),1/math.sqrt(2)]
q1 = [1/math.sqrt(2),-1/math.sqrt(2)]


waveletim = asc.hnm.wavelet2D(im,p1,q1,minLvl=8)
plt.imshow(np.array(waveletim),cmap='gray')
plt.show()

iwaveletim = asc.hnm.iwavelet2D(waveletim,p1,q1,minLvl=8)
plt.subplot(2,1,1)
plt.imshow(np.array(iwaveletim),cmap='gray')
plt.subplot(2,1,2)
plt.imshow(np.array(im),cmap='gray')
plt.show()