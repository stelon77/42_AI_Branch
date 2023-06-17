from eval_formula import *
import itertools


def find_number_variable(formula: str) -> list:
    """
    Take a reverse polish formula and extract
    and ordonnate the variables as a list
    """
    variables = []
    for c in formula:
        if c.isupper():
            variables.append(c)
    return sorted(list(set(variables)))


def printFirstLines(variables: list, n: int) -> None:
    """
    Print the first lines of the table
    """
    if not variables or not n:
        errorMessage()
    first = "| "
    for variable in variables:
        first = first + variable + " | "
    first = first + "= |"
    print(first)
    print("|---" * (n + 1) + "|")


def printTruth(formula: str, variables: list, table: list) -> None:
    """
    Complete the table
    """
    for rank in table:
        n = len(rank)
        line = ""
        changedFormula = formula
        for i in range(n):
            line = line + "| " + str(rank[i]) + " "
            changedFormula = changedFormula.replace(variables[i], str(rank[i]))
        eval = eval_formula(changedFormula)
        if eval is None:
            return errorMessage()
        line = line + "| " + str(int(eval)) + " |"
        print(line)


def print_truth_table(formula: str) -> None:
    """
    a function that takes as input a string
    that contains a propositional formula in reverse polish notation,
    and writes its truth table on the standard output
    """
    variables = find_number_variable(formula)
    n = len(variables)
    printFirstLines(variables, n)
    table = list(itertools.product([0, 1], repeat=n))
    printTruth(formula, variables, table)


def main():
    # print_truth_table("AB&C|")
    # print()
    # print_truth_table("PQRT&>|")
    # # https: // medium.com/street-science/how-to-implement-a-truth-table-generator-in -python-40185e196a5b
    # print()
    # print_truth_table("AB|C&D!&")
    # print()
    # print_truth_table("D!F|D!E!G||H!J|H!I|I!A|G!I!B||A!C|B!C!|&&&&&&&")
    print_truth_table("AB^")
    print()
    print_truth_table("AB|AB&!&")


if __name__ == "__main__":
    main()
