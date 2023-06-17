import numpy as np
from collections.abc import Iterable


class NumPyCreator:
    """
    A class which creates a Numpy array out of other structures
    """
    @staticmethod
    def checkType(datas, dataType):
        if not isinstance(datas, dataType):
            return False
        if isinstance(datas[0], dataType):
            length = len(datas[0])
            for data in datas:
                if (len(data) != length) or (not isinstance(data, dataType)):
                    return False
        return True

    @staticmethod
    def from_list(lst):
        if NumPyCreator.checkType(lst, list):
            return (np.array(lst))
        return None

    @staticmethod
    def from_tuple(tpl):
        if NumPyCreator.checkType(tpl, tuple):
            return (np.array(tpl))
        return None

    @staticmethod
    def from_iterable(itr):
        if NumPyCreator.checkType(itr, Iterable):
            return (np.array(itr))
        return None

    @staticmethod
    def from_shape(shape, value=0.):
        if NumPyCreator.checkType(shape, tuple):
            for nb in shape:
                if not isinstance(nb, int) or nb < 0:
                    return None
            return (np.full(shape, value))
        return None

    @staticmethod
    def random(shape):
        if NumPyCreator.checkType(shape, tuple):
            for nb in shape:
                if not isinstance(nb, int) or nb < 0:
                    return None
            return (np.random.random(shape))
        return None

    @staticmethod
    def identity(n):
        if isinstance(n, int) and n > 0:
            return np.identity(n)
        return None


if __name__ == "__main__":
    from NumPyCreator import NumPyCreator
    npc = NumPyCreator()
    print(repr(npc.from_list([[1, 2, 3], [6, 3, 4]])))
    print(repr(npc.from_list([[1, 2, 3], [6, 4]])))
    print(repr(npc.from_list([[1, 2, 3], ['a', 'b', 'c'], [6, 4, 7]])))
    print(repr(npc.from_list(((1, 2), (3, 4)))))
    print(repr(npc.from_tuple(("a", "b", "c"))))
    print(repr(npc.from_tuple([[1, 2, 3], [6, 3, 4]])))
    print(repr(npc.from_iterable(range(5))))
    shape = (3, 5)
    print(repr(npc.from_shape(shape)))
    print(repr(npc.random(shape)))
    print(repr(npc.identity(4)))
