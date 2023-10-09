import AlgScComp as asc

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

scaling, translation = G.getParametrisation('B')

print(scaling)
print(translation)

print(asc.sfc.parametriseSfc(.999999,scaling,translation,20))
