"""
vector-vector and vector-scalar operations
"""


def checkInput(arg):
    """a function to test argument validity"""
    try:
        if arg is None:
            raise(ValueError)
        if type(arg) is int:
            if arg <= 0:
                raise(ValueError)
        elif type(arg) is tuple:
            if len(arg) != 2:
                raise(ValueError)
            if arg[0] == arg[1]:
                raise(ValueError)
            for nb in arg:
                if type(nb) is not int:
                    raise(ValueError)
        elif type(arg) is list and isinstance(arg[0], (int, float)):
            for nb in arg:
                if not isinstance(nb, (int, float)):
                    raise(ValueError)
        elif type(arg) is list and type(arg[0]) is list:
            for lst in arg:
                if len(lst) != 1:
                    raise(ValueError)
                for nb in lst:
                    if not isinstance(nb, (int, float)):
                        raise(ValueError)
        else:
            raise(ValueError)

    except(ValueError):
        print("The argument of vector is not valid")
        exit()


class Vector:
    """
    Create vector and perform mathematical operations with them

    param :
    list of floats = row vector
    list of list of floats = column vector
    int = size of column vector initialized
    range = column vector with ranged values
    """

    def __init__(self, arg):
        checkInput(arg)
        self.values = []
        if type(arg) is int:
            self.shape = (arg, 1)
            for i in range(0, arg):
                self.values.append([float(i)])
        elif type(arg) is tuple:
            if arg[1] > arg[0]:
                self.shape = (arg[1] - arg[0], 1)
                for i in range(arg[0], arg[1]):
                    self.values.append([float(i)])
            else:
                self.shape = (arg[0] - arg[1], 1)
                for i in range(arg[0], arg[1], -1):
                    self.values.append([float(i)])
        elif type(arg) is list and type(arg[0]) is not list:
            self.shape = (1, len(arg))
            self.values = arg
        else:
            self.shape = (len(arg), 1)
            self.values = arg

    def __add__(self, v2):
        """addition of 2 vectors of the same size"""
        if (not isinstance(v2, Vector)) or (self.shape != v2.shape):
            raise ValueError("addition is only between 2 \
vectors of the same size")

        lst = []
        if self.shape[0] == 1:
            for a, b in zip(self.values, v2.values):
                lst.append(a + b)
        else:
            for i in range(self.shape[0]):
                lst.append([self.values[i][0] + v2.values[i][0]])
        return Vector(lst)

    def __radd__(self, v2):
        return(self.__add__(v2))

    def __sub__(self, v2):
        """substraction of 2 vectors of the same size"""
        if (not isinstance(v2, Vector)) or (self.shape != v2.shape):
            raise ValueError("substraction is only between \
2 vectors of the same size")

        lst = []
        if self.shape[0] == 1:
            for a, b in zip(self.values, v2.values):
                lst.append(a - b)
        else:
            for i in range(self.shape[0]):
                lst.append([self.values[i][0] - v2.values[i][0]])
        return Vector(lst)

    def __rsub__(self, v2):
        return(self.__sub__(v2))

    def __truediv__(self, nb):
        """division of a vector by a scalar"""
        if not isinstance(nb, (int, float)):
            raise ValueError("division only by int or float")
        if nb == 0:
            raise ValueError("division by 0 is impossible")

        lst = []
        if self.shape[0] == 1:
            for a in self.values:
                lst.append(a / nb)
        else:
            for i in range(self.shape[0]):
                lst.append([self.values[i][0] / nb])
        return Vector(lst)

    def __rtruediv__(self, nb):
        raise ValueError("impossible to divide a scalar by a vector")

    def __mul__(self, nb):
        """multiplication of a vector by a scalar"""
        if not isinstance(nb, (int, float)):
            raise ValueError("division only by int or float")
        lst = []
        if self.shape[0] == 1:
            for a in self.values:
                lst.append(a * nb)
        else:
            for i in range(self.shape[0]):
                lst.append([self.values[i][0] * nb])
        return Vector(lst)

    def __rmul__(self, nb):
        return(self.__mul__(nb))

    def __repr__(self):
        """for debugging, on the console"""
        return "Vector({})".format(self.values)

    def __str__(self):
        """A nice description of the vector"""
        ret = "Vector with {} row(s) and {} column(s)\nHere are the values:\n"\
              .format(self.shape[0], self.shape[1])
        if self.shape[0] == 1:
            ret += "[ "
            ret += "  ".join([str(j) for j in self.values]) + " ]\n"
        else:
            for i in range(self.shape[0]):
                ret += str(self.values[i]) + "\n"
        return ret

    def dot(self, v2):
        """multiplication of 2 vectors of the same size gives a scalar"""
        if (not isinstance(v2, Vector)) or (self.shape != v2.shape):
            raise ValueError("dot() is only between vectors of the same size")
        ret = 0
        if self.shape[0] == 1:
            for a, b in zip(self.values, v2.values):
                ret += a * b
        else:
            for i in range(self.shape[0]):
                ret += self.values[i][0] * v2.values[i][0]
        return ret

    def T(self):
        if self.shape[0] == 1:
            list1 = []
            for i in range(self.shape[1]):
                list2 = [self.values[i]]
                list1.append(list2)
        elif self.shape[1] == 1:
            list1 = []
            for i in range(self.shape[0]):
                list1.append(self.values[i][0])
        else:
            raise ValueError("This is not a vector")
        return Vector(list1)
