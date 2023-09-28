"""
Space Filling Curves
====================

This module contains the following algporithms: 

grammar(nonterminals, grammar) - Grammar class for Space Filling Curves 
drawSfc(initialNonTerminal, Grammar, level) - Draws the Space Filling Curve for a given grammar and number of levels
parametriseSfc(x,scalingMatrices,translationVectors,maxItterations) - Returns the (0,1)x(0,1) coorfinates of the parametrised Space Filling Curve at point x in (0,1)

"""
import math
import matplotlib.pyplot as plt

__all__ = ['Grammar','drawSfc','parametriseSfc']

class Grammar:
    '''
    Grammar
    =======

    Class for the construction grammar of the space filling curves.

    Parameters:
    -----------
    nonTerminals: list
        List of strings e.g. ['A','B','C','D']. Can be chosen arbitrarily
    terminals: list 
        List of strings containing the direction. Valid terminals are l, r, u, d (right, left, up and down)
    grammar: list
        List of lists containing one construction grammar for each non terminal. Number of grammars has to be equal
        to the number of non terminals. Each grammar starts and ends with a non terminal.
        e.g. ['A','u','B',...,'D','l','C']

    Raises:
    -------
    Incomplete construction grammar 
    '''
    def __init__(self,nonTerminals: list, terminals: list, grammar: list):
        self.nonTerminals = nonTerminals
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


def recursiveDrawSfc(currentNonTerminal: str,Grammar: Grammar,level: int,currentLevel: int,pointArrayX: list,pointArrayY: list):
    '''
    recursiveDrawSfc
    ================

    Recursive helper function for drawSfc function. Recursively draws the given construction grammar from the terminal statements. 

    Parameters:
    -----------
    currentNonTerminal: str
        Current non terminal statement, for which the current construction grammar is analysed. 
    Grammar: Grammar 
        The grammar structure contains the construction grammar for the space filling curve. 
        The grammar contains the non terinals l, r, u, d (right, left, up and down). The non terminals {A,B,C,...}, 
        Can be chosen arbitrarily. For each non terminal the grammar has to contain the construction grammar in the subsequent layer
    level: int
        Refinemenet level of the space filling curve. Given integer.
    currentLevel: int
        Level of current nonTerminal statement. 
    pointArrayX: list
        List of x coordinates of the space filling curve.
    pointArrayY: list
        List of y coordinates of the space filling curve.
        
    Returns:
    --------
    pointArrayX: list
        List of x coordinates of the space filling curve.
    pointArrayY: list
        List of y coordinates of the space filling curve.
    '''

    currentGrammar = Grammar.grammar[Grammar.nonTerminals.index(currentNonTerminal)]
    if level > currentLevel:
        for i in range(0,len(currentGrammar),2):
            pointArrayX,pointArrayY = recursiveDrawSfc(currentGrammar[i],Grammar,level,currentLevel+1,pointArrayX,pointArrayY)
            factor = 1/(math.sqrt((len(currentGrammar)+1)/2))**(level)
            if i < len(currentGrammar)-1:
                if currentGrammar[i+1] == 'u':
                    pointArrayX.append(pointArrayX[-1])
                    pointArrayY.append(pointArrayY[-1]+factor)
                elif currentGrammar[i+1] == 'd':
                    pointArrayX.append(pointArrayX[-1])
                    pointArrayY.append(pointArrayY[-1]-factor)
                elif currentGrammar[i+1] == 'l':
                    pointArrayX.append(pointArrayX[-1]-factor)
                    pointArrayY.append(pointArrayY[-1])
                elif currentGrammar[i+1] == 'r':
                    pointArrayX.append(pointArrayX[-1]+factor)
                    pointArrayY.append(pointArrayY[-1])
        return pointArrayX,pointArrayY
    else:
        factor = 1/(math.sqrt((len(currentGrammar)+1)/2))**(level)
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
        return pointArrayX,pointArrayY

'''
Callable Functions:

These functions are available from outside the module.
'''

def drawSfc(initialNonTerminal: str, Grammar: Grammar, level: int):
    '''
    drawSfc
    =======

    This function draws the space filling curve (sfc) for a given grammar. The plotting domain is always specified
    to be in (0,1)x(0,1), regardless of the initial plotting pattern. 

    Parameters:
    -----------
    initialNonTerminal: str
        Initial non terminal statement, which shall be plotted, has to be contained in the construction grammar.
    Grammar: Grammar 
        The grammar structure contains the construction grammar for the space filling curve. 
        The grammar contains the non terinals l, r, u, d (right, left, up and down). The non terminals {A,B,C,...}, 
        Can be chosen arbitrarily. For each non terminal the grammar has to contain the construction grammar in the subsequent layer
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
    initGrammar = Grammar.grammar[Grammar.nonTerminals.index(initialNonTerminal)]
    factor = 1/(math.sqrt((len(initGrammar)+1)/2))**(level)
    
    vert = False
    horr = False
    i = 1

    while (vert == False) or (horr == False):
        if initGrammar[i] == 'u':
            factorvert = factor
            vert = True
        elif initGrammar[i] == 'd':
            factorvert = 1-factor
            vert = True
        elif initGrammar[i] == 'l':
            factorhorr = 1-factor
            horr = True
        elif initGrammar[i] == 'r':
            factorhorr = factor
            horr = True
        i = i+2

    pointArrayX, pointArrayY = recursiveDrawSfc(initialNonTerminal,Grammar,level,1,[factorhorr],[factorvert])
    plt.figure
    plt.plot(pointArrayX,pointArrayY)
    plt.show()


def parametriseSfc(x,scalingMatrices,translationVectors,maxItterations):
    '''
    parametriseSfc
    ==============

    This function returns the 2D parametrisation (0,1)x(0,1) for a 1D parameter (0,1) to
    according to the space filling curve logic. The transforms of the given construction grammer have to be given. 

    Parameters:
    -----------
    x: int
        Parameter of evaluation contained in (0,1)
    scalingMatrices: list 
        List of scaling matrices. Ordered in correct parametrisation order. From index 0 to n
    translationVectors: list
        List of translation vectors. Ordered in correct parametrisation order. From index 0 to n
    maxItterations: int 
        Number of maximal itterations. 

    Returns:
    --------
    point: list
        Resulting point in (0,1)x(0,1). 

    Raises:
    -------
    ValueError:
        Parameter value out of bounds (0,1)
    '''

    if x > 1 or x < 0:
        raise ValueError('Parameter value x is only allowed to be in (0,1)')
    if len(scalingMatrices) != len(translationVectors):
        raise ValueError('Given SFC parametrisation not valid. Different number of matrices and vectors.')

    numMatrices = len(scalingMatrices)
    operandList = []
    point = [0,0]
    itt = 1
    
    def calcIndex(x):
        Index = math.floor(x*numMatrices)
        Remainder = x*numMatrices-Index
        return Index, Remainder

    while itt < maxItterations and x != 0:
        Index, Remainder = calcIndex(x)
        x = Remainder
        operandList.append(Index)
        itt += 1

    for i in range(len(operandList)-1,-1,-1):
        Index = operandList[i]
        point0 = scalingMatrices[Index][0][0]*point[0] + scalingMatrices[Index][0][1]*point[1] + translationVectors[Index][0]
        point1 = scalingMatrices[Index][1][0]*point[0] + scalingMatrices[Index][1][1]*point[1] + translationVectors[Index][1]
        point[0] = point0
        point[1] = point1

    return point