from boolean_utilities import *


def conjunctive_normal_form(formula: str) -> str:
    """
    a function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    and returns an equivalent formula in Conjunctive Normal Form (CNF)
    """
    tree = SyntaxTree()
    root = nnfForCnf(formula)
    tree.pushDisjunctionDown(root)
    tree.rearrangeDisjunction(root)
    tree.rearrangeConjunction(root)

    output = []
    tree.postorder(root, output)
    return ''.join(output)


def main():
    # print(conjunctive_normal_form("AB&!"))
    # print(conjunctive_normal_form("AB|!"))
    # print(conjunctive_normal_form("AB|C&"))
    # print(conjunctive_normal_form("AB|C|D|"))
    # print(conjunctive_normal_form("AB&C&D&"))
    # print(conjunctive_normal_form("AB&!C!|"))
    # print(conjunctive_normal_form("AB|!C!&"))
    # print(conjunctive_normal_form("ABCDE&|&|"))
    print(conjunctive_normal_form("ABCD&|&"))
    print("verification")
    print("input truthtable")
    print_truth_table("ABCD&|&")
    print("output truth table")
    print_truth_table(conjunctive_normal_form("ABCD&|&"))


if __name__ == "__main__":
    main()
