from boolean_utilities import *
import time
from random import randint
import os


def log(fonction):
    def logging(*args, **kwargs):
        str2 = fonction.__name__.replace('_', ' ').title()
        timeBefore = time.time()
        ret = fonction(*args, **kwargs)
        timeAfter = time.time()
        execTime = timeAfter - timeBefore
        if execTime < 0.001:
            timeStr = "{:.3f}".format(execTime * 1000) + " ms ]"
        else:
            timeStr = "{:.3f}".format(execTime) + " s ]"
        logLine = "(" + os.environ["USER"] + ")Running: " + \
                  "{: <17}".format(str2) + "[ exec-time = " + timeStr
        # print(logLine)
        with open("machine.log", "a") as logger:
            logger.write(logLine + "\n")
        return ret
    return logging


@log
def SATTruthtable(formula: str, verbose=False) -> bool:
    variables = find_number_variable(formula)
    n = len(variables)
    table = list(itertools.product([0, 1], repeat=n))
    for rank in table:
        changedFormula = formula
        for i in range(n):
            changedFormula = changedFormula.replace(variables[i], str(rank[i]))
        eval = eval_formula(changedFormula)
        if eval is True:
            if verbose is True:
                for i, variable in enumerate(variables):
                    print("{}: {} ".format(variable, bool(rank[i])), end=" ")
            print()
            return True
        if eval is None:
            return errorMessage()
    return False


class SATInstance:
    """
    A class for each formula
    """
    def __init__(self):
        self.variables = []  # la liste des variables
        self.variablesTable = {}  # dictionnaire avec variable-value
        self.clauses = []  # la liste des clauses,avec les valeurs
        self.F = []  # la liste des clauses qui va se modifier

    def parseAndAddClauses(self, formula: str) -> None:
        """
        Parse the formula and separate the different clauses
        Each variable of the dictionnary is value * 2,
        or value * 2 + 1 if the variable is with !
        """
        negated = 0
        formule = formula[::-1]
        nbOfClauses = 1
        nbOflit = 1
        i = 0
        clause = []
        while formule[i] == "&":
            nbOfClauses += 1
            i += 1
        while i < len(formule):
            if formule[i] == "|":
                nbOflit += 1
            elif formule[i] == "!":
                negated = 1
            else:
                variable = formule[i]
                if variable not in self.variablesTable:
                    self.variablesTable[variable] = len(self.variables)
                    self.variables.append(variable)
                encodedLit = self.variablesTable[variable] << 1 | negated
                clause.append(encodedLit)
                nbOflit = nbOflit - 1
                negated = 0
                if nbOflit == 0:
                    self.clauses.append(list(set(clause)))
                    nbOflit = 1
                    clause = []
            i += 1
        if len(self.clauses) != nbOfClauses:  # juste une verif
            print("Ohoh, we made a mistake")
        self.F = self.clauses.copy()

    def literalToString(self, literal: int) -> str:
        """
        Recover the string from numerical variable
        """
        s = '!' if literal & 1 else ''
        return self.variables[literal >> 1] + s

    def clauseToString(self, clause: list) -> str:
        """
        Recover the expression of a clause
        """
        return " ".join(self.literalToString(lit) for lit in clause)

    def propagateUnits(self, F: list) -> list:
        """
        for each unit clause {+/-x} in F
        remove all non-unit clauses containing + /-x
        remove all instances of - /+x in every clause // flipped sign!
        """
        unitList = []
        for clause in F:
            # if len(clause) == 0:
            #     return [[]]
            if len(clause) == 0:
                return [[]]
            if len(clause) == 1:
                unitList.append(clause[0])
        for nb in unitList:
            # if nb ^ 1 in unitList:  # l'inverse
            #     return [[]]
            if nb ^ 1 in unitList:  # l'inverse
                return [[]]
            for clause in F:
                if nb in clause and len(clause) != 1:
                    F.remove(clause)
                if nb ^ 1 in clause:
                    clause.remove(nb ^ 1)
        return F

    def pureElimination(self, F: list) -> list:
        """
        for each variable x
        if +/-x is pure in F
        remove all clauses containing + /-x
        add a unit clause {+/-x}
        """
        lst = []
        for clause in F:
            lst += clause
        uniques = set(lst)
        # print(uniques)
        for val in uniques:
            if val ^ 1 not in uniques:  # means val is pure
                # print(val)
                for clause in F:
                    if val in clause:
                        F.remove(clause)
                F.append([val])
        return F

    def addEliminatedVariables(self, F: list) -> list:
        """
        create a unit clause for eliminated variables during procedure
        """
        f = []
        for clause in F:
            f += clause
        f = set(f)
        for value in self.variablesTable.values():
            a = value << 1
            b = value << 1 | 1
            if a not in f and b not in f:
                F.append([value << 1])
        return F

    def solve(self, F) -> list:
        """
        recursive DPLL algorithm
        """
        F = self.propagateUnits(F)
        F = self.pureElimination(F)
        F = self.addEliminatedVariables(F)
        valueSet = set()
        for clause in F:
            if len(clause) == 0:
                return [[]]
            if len(clause) == 1:
                valueSet.add(clause[0])
        for n in valueSet:
            if n ^ 1 in valueSet:
                return [[]]
        if len(valueSet) == len(self.variables):
            return F
        x = -1
        for clause in F:
            if len(clause) > 1:
                x = clause[0]
        a = self.solve(F + [[x]])
        if a != [[]]:
            return a
        else:
            return self.solve(F + [[x ^ 1]])


@log
def sat(formula: str, verbose=False) -> bool:
    """
    DPLL sat solver, option verbose gives a solution if True
    """
    cnfFormula = conjunctive_normal_form(formula)
    s = SATInstance()
    s.parseAndAddClauses(cnfFormula)
    F = s.clauses.copy()
    F = s.solve(F)
    if F == [[]]:
        return False
    if verbose is True:
        values = []
        for clause in F:
            values += clause
        val = set(values)
        for item in s.variablesTable.items():
            if item[1] << 1 in val:
                print("{}: True,".format(item[0]), end=" ")
            elif item[1] << 1 | 1 in val:
                print("{}: False,".format(item[0]), end=" ")
            else:
                print("y'a pb")
        print()
    return True


def main():
    print("AB|")
    print(SATTruthtable("AB|", True))
    print(sat("AB|", True))
    print()

    print("AB&")
    print(SATTruthtable("AB&", True))
    print(sat("AB&", True))
    print()

    print("AA!&")
    print(SATTruthtable("AA!&", True))
    print(sat("AA!&", True))
    print()

    print("AA^")
    print(SATTruthtable("AA^", True))
    print(sat("AA^", True))
    print()

    print("D!F|D!E!G||H!J|H!I|I!A|G!I!B||A!C|B!C!|&&&&&&&")
    print(SATTruthtable("D!F|D!E!G||H!J|H!I|I!A|G!I!B||A!C|B!C!|&&&&&&&",
                        True))
    print(sat("D!F|D!E!G||H!J|H!I|I!A|G!I!B||A!C|B!C!|&&&&&&&", True))
    print()

    print("ABCDEFGHIJK!LMNOP&&&&&&&&&&&&&&&")
    print(SATTruthtable("ABCDEFGHIJK!LMNOP&&&&&&&&&&&&&&&", True))
    print(sat("ABCDEFGHIJK!LMNOP&&&&&&&&&&&&&&&", True))
    print()


if __name__ == "__main__":
    main()
