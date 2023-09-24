"""
Space Filling Curves
====================

This module contains the following algporithms: 

grammar(nonterminals, grammar) - Grammar class for Space Filling Curves 

"""
import math
import matplotlib.pyplot as plt

__all__ = ['Grammar','drawSfc']

def recursiveDrawSfc(currentNonTerminal,Grammar,level,currentLevel,pointArrayX,pointArrayY):
    currentGrammar = Grammar.grammar[Grammar.nonTerminals.index(currentNonTerminal)]
    if level > currentLevel:
        for i in range(0,len(currentGrammar),2):
            pointArrayX,pointArrayY = recursiveDrawSfc(currentGrammar[i],Grammar,level,currentLevel+1,pointArrayX,pointArrayY)
        return pointArrayX,pointArrayY
    else:
        factor = 1/(math.sqrt(len(currentGrammar)/2))**currentLevel
        for i in range(1,len(currentGrammar),2):
            if currentGrammar[i] == 'u':
                pointArrayX.append(pointArrayX[-1])
                pointArrayY.append(pointArrayY[-1]+factor)
            elif currentGrammar[i] == 'd':
                pointArrayX.append(pointArrayX[-1])
                pointArrayY.append(pointArrayY[-1]-factor)
            elif currentGrammar[i] == 'l':
                pointArrayX.append(pointArrayX[-1]-factor)
                pointArrayY.append(pointArrayY[-1])
            elif currentGrammar[i] == 'r':
                pointArrayX.append(pointArrayX[-1]+factor)
                pointArrayY.append(pointArrayY[-1])

            print(pointArrayX,pointArrayY)
        return pointArrayX,pointArrayY

class Grammar:
    '''
    Grammar
    =======

    Class for the construction grammar of the space filling curves.
    '''
    def __init__(self,nonTerminals, terminals, grammar):
        self.nonTerminals = nonTerminals
        # Parse input 
        if len(grammar) != len(nonTerminals):
            raise ValueError('Grammar not valid, number of construction grammars has to be equal to nonterminals')
        else:
            for i in range(len(nonTerminals)):
                if len(grammar[i])%2 != 1:
                    raise ValueError('Grammar ',i,' has incorrect length.')
                for j in range(len(grammar[i])):
                    if j%2 == 0:
                        if(grammar[i][j] in terminals):
                            raise ValueError('Order of terminals and nonTerminals in grammar ',i,' not correct.')
                    else:
                        if(grammar[i][j] in nonTerminals):
                            raise ValueError('Order of terminals and nonTerminals in grammar ',i,'not correct')
            self.grammar = grammar


def drawSfc(initialNonTerminal, Grammar, level):
    '''
    drawSfc
    =======

    This function draws the space filling curve (sfc) for a given grammar. 

    Parameters:
    -----------
    initialNonTerminal: str
        Initial non terminal statement, which shall be plotted, has to be contained in the construction grammar.
    grammar: Class 
        The grammar structure contains the construction grammar for the space filling curve.
        The grammar contains the non terinals l, r, u, d (right, left, up and down)
        The non terminals {A,B,C,...}, Can be chosen arbitrarily
        For each non terminal the grammar has to contain the construction grammar in the subsequent layer
    level: int
        Refinemenet level of the space filling curve. Given integer.
        
    Returns:
    --------
    fig: figure
        The returned figure contains the constructed sfc

    Raises:
    -------
    Incomplete construction grammar 
    '''
    
    pointArrayX, pointArrayY = recursiveDrawSfc(initialNonTerminal,Grammar,level,1,[0],[0])
    plt.figure
    plt.plot(pointArrayX,pointArrayY)
    plt.show()