import pyfma


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

    def scl(self, term2):
        if isinstance(term2, (float, int)):
            return self.__mul__(term2)
        else:
            raise ValueError('scalar multiplication needs int or float')

    def mul_vec(self, vec):
        if not isinstance(vec, Vector):
            raise TypeError("vec should be a vector")
        if vec.shape[1] != 1:
            vec = vec.T()
        return self.__mul__(vec)

    def mul_mat(self, mat):
        if not isinstance(mat, Matrix):
            raise TypeError("mat should be a matrix")
        return self.__mul__(mat)

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

    def dot(self, v2):
        """multiplication of 2 vectors of the same size gives a scalar"""
        if not isinstance(v2, Matrix):
            raise TypeError("dot() is only between vectors")
        if (self.shape != v2.shape):
            if self.shape[0] == v2.shape[1] and v2.shape[0] == self.shape[1]:
                v2 = v2.T()
            else:
                raise ValueError("vectors has to be same size")
        ret = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # ret += self.data[i][j] * v2.data[i][j]
                ret = pyfma.fma(self.data[i][j], v2.data[i][j], ret)
        return ret

    def vectorToMatrix(self):
        return Matrix(self.data)

    def reshape(self):
        return self.T()

    def norm_1(self):
        if self.shape[0] != 1:
            self = self.reshape()
        ret = 0.0
        for val in self.data[0]:
            if val < 0:
                ret -= val
            else:
                ret += val
        return ret

    def norm(self):
        if self.shape[0] != 1:
            self = self.reshape()
        ret = 0.0
        for val in self.data[0]:
            ret += pow(val, 2)
        return pow(ret, 0.5)

    def norm_inf(self):
        if self.shape[0] != 1:
            self = self.reshape()
        ret = 0.0
        for val in self.data[0]:
            if val > ret:
                ret = val
            elif -val > ret:
                ret = -val
        return ret


def linearCombination(vectorList: list, scalarList: list) -> Vector:
    """
    A function that computes a linear combination of the vectors provided,
    using the corresponding scalar coefficients
    """
    if not isinstance(vectorList, list) or not isinstance(scalarList, list) \
       or len(vectorList) != len(scalarList):
        raise TypeError("param : list of vectors, list of scalars, same size")
    for val in scalarList:
        if not isinstance(val, (int, float, complex)):
            raise ValueError("2nd param: scalar is int or float")
    for v in vectorList:
        if not isinstance(v, Vector):
            raise TypeError("1st param: list of vectors of same dimensions")

    reshaping = False
    shape = vectorList[0].shape
    if vectorList[0].shape[0] != 1:
        reshaping = True  # reshaping as horizontal vector
        shape = vectorList[0].T().shape
    res = Vector(shape)
    for i, v in enumerate(vectorList):
        if reshaping:
            v = v.T()
        if v.shape != shape:
            raise ValueError("all vectors should be of same dimensions")
        # for j, value in enumerate(v.data[0]):
        #     res.data[0][j] = pyfma.fma(value, scalarList[i], res.data[0][j])
        res = (v * scalarList[i]) + res  # equivalent a fma sur matrices
    return res.T()  # vertical vector


def lerp(u, v, t):
    """
    A function that computes a linear interpolation
    between two objects of the same type
    """
    if not isinstance(t, (int, float)) or t < 0 or t > 1:
        raise ValueError("t between 0 and 1")
    if type(u) != type(v):
        raise TypeError("u and v should be of the same type")
    diff = v - u
    return (diff * t) + u


def angle_cos(u, v):
    """
    A functions that compute the cosine of the angle between two given vectors
    Cos(x, y) = x . y / || x || * || y ||
    """
    if isinstance(u, Matrix):
        u = u.matrixToVector()
    a = u.dot(v)
    b = u.norm() * v.norm()
    if b == 0:
        raise ZeroDivisionError("One of the vectors is null")
    return a / b


def cross_product(u, v):
    if not isinstance(u, Matrix) or not isinstance(v, Matrix):
        raise TypeError("u and v should be vectors")
    if u.shape != (1, 3) and u.shape != (3, 1):
        raise ValueError("u should be a 3 dimension vector")
    if v.shape != (1, 3) and v.shape != (3, 1):
        raise ValueError("v should be a 3 dimension vector")
    if u.shape == (3, 1):
        u = u.T()
    if v.shape == (3, 1):
        v = v.T()
    new = []
    new.append(u.data[0][1] * v.data[0][2] - u.data[0][2] * v.data[0][1])
    new.append(u.data[0][2] * v.data[0][0] - u.data[0][0] * v.data[0][2])
    new.append(u.data[0][0] * v.data[0][1] - u.data[0][1] * v.data[0][0])
    return Vector(new).T()  # vertical vector
