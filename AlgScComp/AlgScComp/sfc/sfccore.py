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
        '''
        Initialisation of the grammar.
        The given grammar is checked for dimensional inconsistencies, but not for logical errors.
        If given grammar is wrong, also provided results will be wrong. 
        '''
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

    def getParametrisation(self,initialNonterminal: str):
        '''
        getParametrisation
        ==================

        This function returns the scaling matrices and translation vectors of the grammar. These scaling matrices can be used for 
        the parametrisation in 2D. The parametrisation is always starting at the origin (x,y) = (0,0). 

        Parameters:
        -----------
        initialNonterminal: str
            This is the first given nonterminal and the overall shape of the space filling curve 

        Returns:
        --------
        scalingMatrices: list
            Scaling matrices in the order of the provided gramar and suitable for the function parametriseSfc. 
        translationVectors: list
            Translation vectors in the order of the provided grammar and suitable for the function parametriseSfc.

        Raises:
        -------
        None
        '''
        firstMainTerminal, secondMainTerminal = self.__getFirstTerminals(initialNonterminal)
        gram = self.grammar[self.nonTerminals.index(initialNonterminal)]

        factor = 1/math.sqrt((len(gram)+1)/2)

        scalingMatrices = [None] * int((len(gram)+1)/2)
        translationVectors = [None] * int((len(gram)+1)/2)

        vertical = ['u','d']
        horrizontal = ['l','r']

        if firstMainTerminal in vertical:
            firstMainDirection = vertical
        else:
            firstMainDirection = horrizontal

        start = [0,0]

        for i in range(0,len(gram),2):
            currentGram = self.grammar[self.nonTerminals.index(gram[i])]

            translationVectors[int(i/2)] = start.copy()

            for j in range(1,len(currentGram),2):
                if currentGram[j] == 'u':
                    start[1] += factor/(math.sqrt((len(gram)+1)/2)-1)
                elif currentGram[j] == 'd':
                    start[1] -= factor/(math.sqrt((len(gram)+1)/2)-1)
                elif currentGram[j] == 'r':
                    start[0] += factor/(math.sqrt((len(gram)+1)/2)-1)
                elif currentGram[j] == 'l':
                    start[0] -= factor/(math.sqrt((len(gram)+1)/2)-1)

            firstTerminal, secondTerminal = self.__getFirstTerminals(gram[i])

            scalingMatrix = [[0,0],[0,0]]

            if firstTerminal in firstMainDirection:
                if firstMainTerminal == firstTerminal:
                    if firstTerminal in horrizontal:
                        scalingMatrix[0][0] = factor
                    else:
                        scalingMatrix[1][1] = factor
                else:
                    if firstTerminal in horrizontal:
                        scalingMatrix[0][0] = -factor
                    else:
                        scalingMatrix[1][1] = -factor
                if secondMainTerminal == secondTerminal:
                    if secondTerminal in horrizontal:
                        scalingMatrix[0][0] = factor
                    else:
                        scalingMatrix[1][1] = factor
                else:
                    if secondTerminal in horrizontal:
                        scalingMatrix[0][0] = -factor
                    else:
                        scalingMatrix[1][1] = -factor
            else:
                if firstMainTerminal == secondTerminal:
                    if firstTerminal in horrizontal:
                        scalingMatrix[0][1] = factor
                    else:
                        scalingMatrix[1][0] = factor
                else:
                    if firstTerminal in horrizontal:
                        scalingMatrix[0][1] = -factor
                    else:
                        scalingMatrix[1][0] = -factor
                if secondMainTerminal == firstTerminal:
                    if secondTerminal in horrizontal:
                        scalingMatrix[0][1] = factor
                    else:
                        scalingMatrix[1][0] = factor
                else:
                    if secondTerminal in horrizontal:
                        scalingMatrix[0][1] = -factor
                    else:
                        scalingMatrix[1][0] = -factor

            scalingMatrices[int(i/2)] = scalingMatrix

        return scalingMatrices, translationVectors

    def __getFirstTerminals(self,currentNonterminal):
        '''
        Non-callable helper function, returning the first terminal in horrizontaland vertical direction. 
        '''
        currentGrammar = self.grammar[self.nonTerminals.index(currentNonterminal)]
        sideLength = math.sqrt((len(currentGrammar)+1)/2)

        firstTerminal = currentGrammar[1]

        vertical = ['u','d']
        horrizontal = ['l','r']

        if firstTerminal in vertical:
            secondDirection = horrizontal
        else:
            secondDirection = vertical

        for i in range(3,int(sideLength*2),2):
            if currentGrammar[i] in secondDirection:
                secondTerminal = currentGrammar[i]
                break

        return firstTerminal, secondTerminal

    
     
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

