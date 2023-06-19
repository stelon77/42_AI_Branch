def errorExit():
    raise TypeError("wrong datas to initialize Matrix")


def errorExit2():
    raise ValueError("wrong datas to initialize Matrix")


def errordata(data, type_data, type_sub_data=None, type_sub_sub_data=None):
    """
    Return True if the datas are the wrong type
    """
    if not isinstance(data, type_data):
        return True
    if type_sub_data is not None:
        for sub in data:
            if not isinstance(sub, type_sub_data):
                return True
            if type_sub_sub_data is not None:
                for sub_sub in sub:
                    if not isinstance(sub_sub, type_sub_sub_data):
                        return True
    return False


class Matrix:
    def __init__(self, *args):
        if len(args) != 1:
            errorExit2()
        data = args[0]
        if errordata(data, list, list, (float, int)) \
           and errordata(data, tuple, int):
            errorExit()
        if isinstance(data, tuple):
            if len(data) != 2 or data[0] <= 0 or data[1] <= 0:
                errorExit2()
            self.data = []
            for row in range(data[0]):
                column = []
                for n in range(data[1]):
                    column.append(0.)
                self.data.append(column)
            self.shape = data
        else:
            self.data = data
            shape_1 = len(data[0])
            if shape_1 == 0:
                errorExit()
            shape_0 = 0
            for index_row, row in enumerate(data):
                if len(row) != shape_1:
                    errorExit()
                for index_column, number in enumerate(row):
                    self.data[index_row][index_column] = float(number)
                shape_0 += 1
            self.shape = (shape_0, shape_1)

    def __add__(self, m2):
        if not isinstance(m2, Matrix) or self.shape != m2.shape:
            raise ArithmeticError("addition only between \
two matrices of same dimensions")
            return None
        lst = []
        for i in range(self.shape[0]):
            lst2 = []
            for j in range(self.shape[1]):
                lst2.append(self.data[i][j] + m2.data[i][j])
            lst.append(lst2)
        return (type(self)(lst))

    def __radd__(self, m2):
        return self.__add__(m2)

    def __sub__(self, m2):
        if not isinstance(m2, Matrix) or self.shape != m2.shape:
            raise ArithmeticError("substraction only \
between two matrices of same dimensions")
        lst = []
        for i in range(self.shape[0]):
            lst2 = []
            for j in range(self.shape[1]):
                lst2.append(self.data[i][j] - m2.data[i][j])
            lst.append(lst2)
        return (type(self)(lst))

    def __rsub__(self, m2):
        return self.__sub__(m2)

    def __truediv__(self, nb):
        if isinstance(nb, (Matrix, Vector)):
            if nb.shape != (1, 1):
                raise ArithmeticError("division only by 1x1 matrix or vector")
            number = nb.data[0][0]
        elif isinstance(nb, (float, int)):
            number = nb
        else:
            raise ArithmeticError("division only by int or float")
        if number == 0:
            raise ZeroDivisionError("division by 0 is impossible")
        lst = []
        for i in range(self.shape[0]):
            lst2 = []
            for j in range(self.shape[1]):
                lst2.append(self.data[i][j] / number)
            lst.append(lst2)
        return (type(self)(lst))

    def __rtruediv__(self, nb):
        if not isinstance(nb, (int, float, Matrix, Vector)):
            raise TypeError("first term cannot be divided")
        if self.shape == (1, 1):
            number = self.data[0][0]
            lst = [[nb / number]]
            return Vector(lst)
        raise ValueError("scalar can only be divided by 1x1 vector or matrix")

    def __mul__(self, term2):
        if not isinstance(term2, (float, int, Vector, Matrix)):
            raise ArithmeticError("wrong term in multiplcatiom")
        lst = []
        if isinstance(term2, (float, int)):
            for i in range(self.shape[0]):
                lst2 = []
                for j in range(self.shape[1]):
                    lst2.append(self.data[i][j] * term2)
                lst.append(lst2)
            return (type(self)(lst))

        elif isinstance(term2, (Matrix, Vector)):
            if self.shape[1] != term2.shape[0]:
                raise AttributeError('wrong shape for matrix * matrix \
or matrix * vector')
            for i in range(self.shape[0]):
                lst2 = []
                for j in range(term2.shape[1]):
                    nb = 0.0
                    for k in range(self.shape[1]):
                        nb += self.data[i][k] * term2.data[k][j]
                    lst2.append(nb)
                lst.append(lst2)
            # if isinstance(term2, Vector):
            if len(lst) == 1 or len(lst[0]) == 1:
                return Vector(lst)
            return Matrix(lst)

    def __rmul__(self, term2):
        if isinstance(term2, (float, int)):
            return self.__mul__(term2)
        else:
            raise ValueError('matrix multiplication is not commutative')

    def __str__(self):
        ret = "shape : {}".format(self.shape) + '\n'
        for row in self.data:
            ret += str(row) + '\n'
        return ret

    def __repr__(self):
        return (str(type(self))[15: -2] + '(' + str(self.data) + ')')

    def T(self):
        lst = []
        for j in range(self.shape[1]):
            lst2 = []
            for i in range(self.shape[0]):
                lst2.append(self.data[i][j])
            lst.append(lst2)
        return (type(self)(lst))


class Vector(Matrix):
    def __init__(self, data):
        Matrix.__init__(self, data)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError('This is not a vector')

    def dot(self, v2):
        """multiplication of 2 vectors of the same size gives a scalar"""
        if (not isinstance(v2, Vector)) or (self.shape != v2.shape):
            raise ValueError("dot() is only between vectors of the same size")
        ret = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ret += self.data[i][j] * v2.data[i][j]
        return ret


if __name__ == "__main__":
    ma = Matrix([[1.0, 2.0], [3.0, 4.0], [5, 6]])
    mi = Matrix((4, 3))
    # print(ma.shape)
    # print(ma.data)
    mb = Matrix([[2.0, 1.0], [4.0, 3.0], [6, 5]])
    mc = ma + mb
    print(mc)
    # print(mc.shape)
    # print(mc.data)
    print(mc / 3)
    md = 3 / mc
