import AlgScComp as asc
import random
import matplotlib.pyplot as plt

xvec = [0,1,2,3,4,6,5,7,8,9]
yvec = [random.random() for i in range(0,10)]

xint = [0, 1, 2.1, 3.8, 4.3, 5.1]
yint = asc.hnm.plinint(xvec,yvec,xint)

plt.figure()
plt.plot(xvec,yvec)
plt.scatter(xint,yint)
plt.show()


S = [(0,1),(0.2,1),(0.3,-1),(0.35,1),(0.4,1),(0.55,-1),(0.6,-1),(0.65,-1),(0.7,-1),(1,1)]
#vector with 1D features
xVec = [s[0] for s in S]
#vector with labels of the training data
yVec = [s[1] for s in S]

C = asc.hnm.Classifier1Dnodal(3)
C.train(xVec,yVec)
print(C.classify([0.5,0.7]))

