import AlgScComp as asc
import matplotlib.pyplot as plt
import numpy as np

nonTerminals = ['P','Q','R','S']
terminals = ['l','r','u','d']
grammar = [
    ['P','u','Q','u','P','r','R','d','S','d','R','r','P','u','Q','u','P'],
    ['Q','u','P','u','Q','l','S','d','R','d','S','l','Q','u','P','u','Q'],
    ['R','d','S','d','R','r','P','u','Q','u','P','r','R','d','S','d','R'],
    ['S','d','R','d','S','l','Q','u','P','u','Q','l','S','d','R','d','S']
]

G = asc.sfc.Grammar(nonTerminals,terminals,grammar)

print(G.nonTerminals)
print(G.grammar)

asc.sfc.drawSfc('S',G,4)

scalingMatrices = [[[0,.5],[.5,0]],[[.5,0],[0,.5]],[[.5,0],[0,.5]],[[0,-.5],[-.5,0]]]
translationVectors = [[0,0],[0,.5],[.5,.5],[1,.5]]

print(asc.sfc.parametriseSfc(1/3,scalingMatrices,translationVectors,20))

nonTerminals = ['H','A','B','C']
terminals = ['l','r','u','d']
grammar = [
    ['A','u','H','r','H','d','B'],
    ['H','r','A','u','A','l','C'],
    ['C','l','B','d','B','r','H'],
    ['B','d','C','l','C','u','A']
]

G = asc.sfc.Grammar(nonTerminals,terminals,grammar)

scaling, translation = G.getParametrisation('H')

print(scaling)
print(translation)

u = np.linspace(0,1,10000).tolist()

x = [None] * (len(u)-2)
y = [None] * (len(u)-2)

for i in range(1,len(x)-1):
    [x[i-1],y[i-1]] = asc.sfc.parametriseSfc(u[i],scaling,translation,100)

plt.figure()
sections = 9
for i in range(sections):
    plt.plot(x[i*int(10000/sections):(i+1)*int(10000/sections)],y[i*int(10000/sections):(i+1)*int(10000/sections)])
plt.plot(x[0],y[0],'ro')
plt.show()
