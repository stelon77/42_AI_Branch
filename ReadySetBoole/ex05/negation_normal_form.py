from boolean_utilities import *


def negation_normal_form(formula: str) -> str:
    """
    A function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    and returns an equivalent formula in Negation Normal Form (NNF)
    """
    tree = SyntaxTree()
    root = postfixToSyntaxTree(formula)

    tree.eliminateImplicationAndEquivalence(root)
    if not root:
        return
    while root.value == "!" and not root.left.value.isupper():
        root = tree.transformInitialNegationNode(root)
    tree.manageNegations(root)

    output = []
    tree.postorder(root, output)
    return ''.join(output)


def main():
    print(negation_normal_form("AB^"))
    print(negation_normal_form("AB&!"))
    print(negation_normal_form("AB|!"))
    print(negation_normal_form("AB>"))
    print(negation_normal_form("AB="))
    print(negation_normal_form("AB|C&!"))
    print(negation_normal_form("AB|C&!&"))


if __name__ == "__main__":
    main()
