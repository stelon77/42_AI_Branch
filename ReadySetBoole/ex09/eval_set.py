from boolean_utilities import *


def createTreeAndCheckDatas(formula: str, sets: set) -> Node:
    """
    Checks the coherence of datas and create a syntax tree with
    nodes including
    sets and
    a memory of all elements in case of negation of the set
    """
    variables = find_number_variable(formula)
    n = len(variables)
    if not checkSets(sets) or len(sets) != n:
        return errorMessage()
    setDict = {}
    allSet = []
    for a, b in zip(variables, sets):
        setDict[a] = b
        allSet = allSet + list(b)
    wholeSet = set(allSet)
    root = postfixToSyntaxTree(formula, setDict, wholeSet)
    if not root:
        return errorMessage()
    return root


def checkSets(sets: set) -> bool:
    """
    Check the validity of the sets input
    """
    if not isinstance(sets, list):
        return False
    for subset in sets:
        if not isinstance(subset, (set, frozenset)):
            return False
        for nb in subset:
            if not isinstance(nb, int):
                return False
    return True


def eval_set(formula: str, sets: set) -> list:
    """
    A function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    and a list of sets(each containing numbers),
    then evaluates this list and returns the resulting set as a list
    """
    tree = SyntaxTree()
    nnfFormula = negation_normal_form(formula)
    root = createTreeAndCheckDatas(nnfFormula, sets)
    if root:
        return list(tree.evaluateSets(root)[0])
    return None


def main():
    lolo = [set([1, 2]), set([3]), set([1, 3, 4])]
    lolo2 = [set([0, 1, 2, 5]), set([0, 3, 4, 5])]
    example1 = [set([0, 1, 2]), set([0, 3, 4])]
    example2 = [set([0, 1, 2]), set([3, 4, 5])]
    example3 = [set([0, 1, 2])]

    print(eval_set("AB^", lolo2))
    print(eval_set("AB&", example1))
    print(eval_set("AB|", example2))
    print(eval_set("A!", example3))
    print(eval_set("A!", lolo2))


if __name__ == "__main__":
    main()
