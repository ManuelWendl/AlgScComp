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

#print(G.nonTerminals)
#print(G.grammar)

asc.sfc.drawSfc('Q',G,2)