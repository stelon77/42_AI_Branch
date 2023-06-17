class Matrix:
    """
    Create matrix and perform mathematical operations with them

    param :
    list of lists of int, floats or complex number
    shape (tuple) : the matrix of this shape wil be filled with zeroes
    """

    def __init__(self, *args):
        """
        attributes :
            shape : shape of the matrix
            data : list of list of the values of the matrix
        """
        if len(args) != 1:
            self.errorExit2()
        data = args[0]
        if self.errordata(data, list, list, (float, int, complex)) \
           and self.errordata(data, tuple, int):
            self.errorExit()
        if isinstance(data, tuple):
            if len(data) != 2 or data[0] <= 0 or data[1] <= 0:
                self.errorExit2()
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
                self.errorExit()
            shape_0 = 0
            for index_row, row in enumerate(data):
                if len(row) != shape_1:
                    self.errorExit()
                for index_column, number in enumerate(row):
                    self.data[index_row][index_column] = float(number)
                    # a revoir pour les nombres complexes
                shape_0 += 1
            self.shape = (shape_0, shape_1)

    def __add__(self, m2):
        """
        A method to add two matrix of same size
        """
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

    def add(self, m2):
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

    def sub(self, m2):
        return self.__sub__(m2)

    def __mul__(self, term2):
        if not isinstance(term2, (float, int)):
            raise ArithmeticError("wrong term in multiplcatiom")
        lst = []
        for i in range(self.shape[0]):
            lst2 = []
            for j in range(self.shape[1]):
                lst2.append(self.data[i][j] * term2)
            lst.append(lst2)
        return (type(self)(lst))

    def __rmul__(self, term2):
        if isinstance(term2, (float, int)):
            return self.__mul__(term2)
        else:
            raise ValueError('matrix multiplication is not commutative')

    def scl(self, term2):
        if isinstance(term2, (float, int)):
            return self.__mul__(term2)
        else:
            raise ValueError('scalar multiplication needs int or float')

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

    def size(self):
        return self.shape

    def isSquare(self):
        return self.shape[0] == self.shape[1]

    def matrixToVector(self):
        if self.shape[0] == 1 or self.shape[1] == 1:
            return Vector(self.data)
        raise ValueError("Wrong shape to be converted in vector")

    @staticmethod
    def errorExit():
        raise TypeError("wrong datas to initialize Matrix")

    @staticmethod
    def errorExit2():
        raise ValueError("wrong datas to initialize Matrix")

    @staticmethod
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


class Vector(Matrix):
    def __init__(self, data):
        """
        accepts also a single list of floats, int or complex
        """
        if isinstance(data, list):
            if not isinstance(data[0], list):
                data2 = []
                data2.append(data)
                data = data2
        Matrix.__init__(self, data)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError('This is not a vector')

    def __add__(self, m2):
        if not isinstance(m2, Matrix):
            raise ValueError("The secund term has to be a matrix/vector")
        if m2.shape[0] == self.shape[1] and m2.shape[1] == self.shape[0]:
            m2 = m2.T()
        return Matrix.__add__(self, m2)

    def __sub__(self, m2):
        if not isinstance(m2, Matrix):
            raise ValueError("The secund term has to be a matrix/vector")
        if m2.shape[0] == self.shape[1] and m2.shape[1] == self.shape[0]:
            m2 = m2.T()
        return Matrix.__sub__(self, m2)

    def vectorToMatrix(self):
        return Matrix(self.data)


if __name__ == "__main__":
    ma = Matrix([[1.0, 2.0], [3.0, 4.0], [5, 6]])
    # mi = Matrix((4, 3))
    print(ma)
    # print(mi)
    # print(str(type(ma)))
    # print(ma.shape)
    # print(ma.isSquare())
    mb = Matrix([[1.0], [2.0]])
    print(str(type(mb)))
    vb = mb.matrixToVector()
    print(str(type(vb)))
    print(vb)
    mc = vb.vectorToMatrix()
    print(str(type(mc)))
    print(mc)
