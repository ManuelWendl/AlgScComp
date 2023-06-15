import AlgScComp as asc
import random
import matplotlib.pyplot as plt
import math

xvec = [0,1,2,3,4,5,6,7,8,9]
yvec = [random.random() for i in range(0,10)]

xint = [0, 1, 2.1, 3.8, 4.3, 5.1]
yint = asc.hnm.plinint(xvec,yvec,xint)

plt.figure()
plt.plot(xvec,yvec)
plt.scatter(xint,yint)
plt.show()


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

C.plotBasis(False)

plt.plot(xVec,yVec,'-ok')
y = [C.classify(i/100) for i in range(0,100)]
x = [i/100 for i in range(0,100)]
plt.plot(x,y,'r')
plt.show()


p1 = [1/math.sqrt(2),1/math.sqrt(2)]
q1 = [1/math.sqrt(2),-1/math.sqrt(2)]

p2 = [(1+math.sqrt(3))/(4*math.sqrt(2)),(3+math.sqrt(3))/(4*math.sqrt(2)),(3-math.sqrt(3))/(4*math.sqrt(2)),(1-math.sqrt(3))/(4*math.sqrt(2))]
q2 =  [(1+math.sqrt(3))/(4*math.sqrt(2)),-(3+math.sqrt(3))/(4*math.sqrt(2)),(3-math.sqrt(3))/(4*math.sqrt(2)),-(1-math.sqrt(3))/(4*math.sqrt(2))]

s2 = [8, 4, -1, 1, 0, 4, 1, 7, -5/2, -3/2, 0, -4, -2, -2, 1, -5]
print(asc.hnm.wavelet1D(s2,p2,q2,edgeTreat='zeros'))
print('Inverse Test')
a = asc.hnm.wavelet1D(s2,p2,q2,edgeTreat='zeros')
print(a)
b= asc.hnm.iwavelet1D(a,p2,q2,edgeTreat='zeros')
print('b=',b)
