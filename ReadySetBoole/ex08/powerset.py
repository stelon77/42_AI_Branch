# from boolean_utilities import *
import itertools


def powerset(aSet: set) -> list:
    """
    a function that takes as input a set of integers, and returns its powerset
    """
    if not isinstance(aSet, set):
        return None
    n = len(aSet)
    result = list()
    if n == 0:
        list2 = list()
        result.append(list2)
        return result
    for value in aSet:
        if not isinstance(value, int):
            return None
    values = list(aSet)
    table = list(itertools.product([0, 1], repeat=n))
    for rank in table:
        list2 = []
        for i, nb in enumerate(rank):
            if nb:
                list2.append(values[i])
        result.append(list2)
    return result


def main():
    myset = set()
    print(powerset(myset))
    myset = {1, -1}
    print(powerset(myset))
    myset = {0, 1, 2}
    print(powerset(myset))
    myset = {0}
    print(powerset(myset))


if __name__ == "__main__":
    main()
