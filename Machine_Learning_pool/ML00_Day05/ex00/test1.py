# These tests have been created by Madvid...Praise to him

from unittest import TestCase
import unittest
import matrix


def check_data_equality(expected, test):
    n_line = len(expected)
    n_col = len(expected[0])
    for ii in range(n_line):
        for jj in range(n_col):
            if expected[ii][jj] != test[ii][jj]:
                return False
    return True


def check_shape(expected, test):
    if expected == test:
        return True
    return False


class TestMatrixMethods(TestCase):
    # Tests group 1
    def test_basic_instance_list_1(self):
        expected_val = [[0, 0], [0, 0]]
        expected_shape = (2,2)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_list_2(self):
        expected_val = [[1, 2], [3, 4]]
        expected_shape = (2,2)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_list_3(self):
        expected_val = [[21], [42]]
        expected_shape = (2,1)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_list_4(self):
        expected_val = [[21, 42, 63]]
        expected_shape = (1,3)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))


    # Tests group 2
    def test_basic_instance_shape_1(self):
        expected_val = [[0, 0], [0, 0]]
        expected_shape = (2,2)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_shape_2(self):
        expected_val = [[0], [0], [0]]
        expected_shape = (3,1)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_shape_3(self):
        expected_val = [[0, 0, 0]]
        expected_shape = (1,3)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_shape_4(self):
        expected_val = [[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]
        expected_shape = (5,6)
        M = matrix.Matrix(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))


    # Tests group 3
    def test_error_instance_1(self):
        # No argument
        with self.assertRaises(ValueError):
            matrix.Matrix()

    def test_error_instance_2(self):
        # None as argument
        with self.assertRaises(TypeError):
            matrix.Matrix(None)

    def test_error_instance_3(self):
        # str as argument
        with self.assertRaises(TypeError):
            matrix.Matrix("toto")

    def test_error_instance_4(self):
        # integer as argument
        with self.assertRaises(TypeError):
            matrix.Matrix(3)

    def test_error_instance_5(self):
        # float as argument
        with self.assertRaises(TypeError):
            matrix.Matrix(3.142)

    def test_error_instance_6(self):
        # complex as argument
        with self.assertRaises(TypeError):
            matrix.Matrix(complex(4, 2))

    def test_error_instance_7(self):
        # double nested list (3 squared brackets) as argument
        with self.assertRaises(TypeError):
            matrix.Matrix([[[1, 2, 3]]])

    def test_error_instance_8(self):
        # Incorrect tuple as argument
        with self.assertRaises(ValueError):
            matrix.Matrix((1, 2, 3))


    # Tests group 4
    def test_addition_1(self):
        # basic (2x2) with (2x2) addition matrix with shape instance method
        m1 = matrix.Matrix((2, 2))
        m2 = matrix.Matrix((2, 2))
        m3 = m1 + m2
        expected_val = [[0, 0], [0, 0]]
        expected_shape = (2, 2)
        self.assertTrue(isinstance(m3, matrix.Matrix) and check_data_equality(expected_val, m3.data) and check_shape(expected_shape, m3.shape))

    def test_addition_2(self):
        # basic (3x3) with (3x3) addition matrix with nested list instance method
        m1 = matrix.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m2 = matrix.Matrix([[1, 1, 1], [2, 3, 2], [5, 4, 5]])
        m3 = m1 + m2
        expected_val = [[2, 3, 4], [6, 8, 8], [12, 12, 14]]
        expected_shape = (3, 3)
        self.assertTrue(isinstance(m3, matrix.Matrix) and check_data_equality(expected_val, m3.data) and check_shape(expected_shape, m3.shape))

    def test_addition_3(self):
        # basic (3x3) with (3x3) addition matrix with nested list instance method. The second matrix has all it component negative
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = matrix.Matrix([[-1, -1, -1, -1], [-2, -3, -2, 10.3]])
        m3 = m1 + m2
        expected_val = [[0, 1, 2, 9.5], [2, 2, 4, 5.1]]
        expected_shape = (2, 4)
        self.assertTrue(isinstance(m3, matrix.Matrix)) \
            and self.assertAlmostEqual(expected_val, m3.data) \
                and self.assertEqual(expected_shape, m3.shape)


    # Tests group 5
    def test_error_addition_1(self):
        # Addition between a (2x4) and a (2x2) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = matrix.Matrix([[-1, -1], [-2, -3]])
        with self.assertRaises(ArithmeticError):
            m1 + m2

    def test_error_addition_2(self):
        # Addition beteen a Matrix and a nested list of the same shape
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = [[0, 0, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ArithmeticError):
            m1 + m2

    def test_error_addition_3(self):
        # Addition between (3x3) with tuple
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = 3246
        with self.assertRaises(ArithmeticError):
            m1 + m2


    # Tests group 6
    def test_substraction_1(self):
        # basic (2x2) with (2x2) substraction matrix with shape instance method
        m1 = matrix.Matrix((2, 2))
        m2 = matrix.Matrix((2, 2))
        m3 = m1 - m2
        expected_val = [[0, 0], [0, 0]]
        expected_shape = (2, 2)
        self.assertTrue(isinstance(m3, matrix.Matrix) and check_data_equality(expected_val, m3.data) and check_shape(expected_shape, m3.shape))

    def test_substraction_2(self):
        # basic (3x3) with (3x3) substraction matrix with nested list instance method
        m1 = matrix.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m2 = matrix.Matrix([[1, 1, 1], [2, 3, 2], [5, 4, 5]])
        m3 = m1 - m2
        expected_val = [[0, 1, 2], [2, 2, 4], [2, 4, 4]]
        expected_shape = (3, 3)
        self.assertTrue(isinstance(m3, matrix.Matrix) and check_data_equality(expected_val, m3.data) and check_shape(expected_shape, m3.shape))

    def test_substraction_3(self):
        # basic (3x3) with (3x3) substraction matrix with nested list instance method. The second matrix has all it component negative
        m1 = matrix.Matrix([[1, 2, 3, 9.5], [4, 5, 6, -4.5]])
        m2 = matrix.Matrix([[-1, -1, -1, 10], [-2, -3, -2, -5]])
        m3 = m1 - m2
        expected_val = [[2, 3, 4, -0.5], [6, 8, 8, 0.5]]
        expected_shape = (2, 4)
        self.assertTrue(isinstance(m3, matrix.Matrix) and check_data_equality(expected_val, m3.data) and check_shape(expected_shape, m3.shape))


    # Tests group 7
    def test_error_substraction_1(self):
        # Substraction between a (2x4) and a (2x2) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = matrix.Matrix([[-1, -1], [-2, -3]])
        with self.assertRaises(ArithmeticError):
            m1 + m2

    def test_error_substraction_2(self):
        # Substraction beteen a Matrix and a nested list of the same shape
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = [[0, 0, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ArithmeticError):
            m1 + m2

    def test_error_substraction_3(self):
        # Substraction between (3x3) with tuple
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = 3246
        with self.assertRaises(ArithmeticError):
            m1 + m2


    # Tests group 8
    def test_multiplication_1(self):
        # basic (2x2) with (2x2) multiplication matrices
        m1 = matrix.Matrix([[1, 2], [3, 4]])
        m2 = matrix.Matrix([[11, 22], [33, 44]])
        expected_val = [[77, 110], [165, 242]]
        expected_shape = (2, 2)
        self.assertTrue(isinstance(m1 * m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 * m2).data) \
                and self.assertEqual(expected_shape, (m1 * m2).shape)

    def test_multiplication_2(self):
        # basic (4x4) with (4x4) multiplication matrices
        m1 = matrix.Matrix([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
        m2 = matrix.Matrix([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
        expected_val = [[1, 0, 0, 0], [0, 4, 0, 0], [0, 0, 9, 0], [0, 0, 0, 16]]
        expected_shape = (4, 4)
        self.assertTrue(isinstance((m1 * m2), matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 * m2).data) \
                and self.assertEqual(expected_shape, (m1 * m2).shape)

    def test_multiplication_3(self):
        # basic (3x2) with (2x3) multiplication matrices
        m1 = matrix.Matrix([[1, 9, 4], [-2, 2, -5]])
        m2 = matrix.Matrix([[3, -7], [0, -3], [8, 0]])
        expected_val = [[35, -34], [-46, 8]]
        expected_shape = (2, 2)
        self.assertTrue(isinstance(m1 * m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 * m2).data) \
                and self.assertEqual(expected_shape, (m1 * m2).shape)

    def test_multiplication_4(self):
        # basic (3x3) multiplication matrix with a float
        m1 = matrix.Matrix([[1, 9, 4], [-2, 2, -5]])
        m2 = 0.5
        expected_val = [[0.5, 4.5, 2.0], [-1.0, 1.0, -2.5]]
        expected_shape = (2, 3)
        self.assertTrue(isinstance(m1 * m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 * m2).data) \
                and self.assertEqual(expected_shape, (m1 * m2).shape)

    def test_multiplication_5(self):
        # basic multiplication of a int with (3x2) multiplication matrix
        m1 = 2
        m2 = matrix.Matrix([[3, -7], [0, -3], [8, 0]])
        expected_val = [[6, -14], [0, -6], [16, 0]]
        expected_shape = (3, 2)
        self.assertTrue(isinstance(m1 * m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 * m2).data) \
                and self.assertEqual(expected_shape, (m1 * m2).shape)


    # Tests group 9
    def test_error_multiplication_1(self):
        # multiplication between a (2x4) Matrix and a (2x2) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = matrix.Matrix([[-1, -1], [-2, -3]])
        with self.assertRaises(AttributeError):
            m1 * m2

    def test_error_multiplication_2(self):
        # multiplication between a (1x3) Matrix and a (2x4) Matrix
        m1 = matrix.Matrix([[-1, -1, -1]])
        m2 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        with self.assertRaises(AttributeError):
            m1 * m2

    def test_error_multiplication_3(self):
        # multiplication between a (2x4) Matrix and a (2x1) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = matrix.Matrix([[1]])
        with self.assertRaises(AttributeError):
            m1 * m2

    def test_error_multiplication_4(self):
        # multiplication between a (2x4) Matrix and a (2x1) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = "Matrix"
        with self.assertRaises(ArithmeticError):
            m1 * m2

    def test_error_multiplication_5(self):
        # multiplication between a (2x4) Matrix and a (2x1) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = (2,5)
        with self.assertRaises(ArithmeticError):
            m1 * m2


    # Tests group 10
    def test_division_1(self):
        # basic (2x2) with (2x2) division matrix with shape instance method
        m1 = matrix.Matrix([[2, 4, 5, 6], [11, -12, 13, -14]])
        m2 = 2
        expected_val = matrix.Matrix([[1, 2, 2.5, 3], [5.5, -6, 6.5, -7]])
        expected_shape = (2, 4)
        self.assertTrue(isinstance(m1 / m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 / m2).data) \
                and self.assertEqual(expected_shape, (m1 / m2).shape)

    def test_division_2(self):
        # basic (3x3) with (3x3) division matrix with nested list instance method
        m1 = matrix.Matrix([[2, 4, 5, 6], [11, -12, 13, -14]])
        m2 = matrix.Matrix([[2]])
        expected_val = matrix.Matrix([[1, 2, 2.5, 3], [5.5, -6, 6.5, -7]])
        expected_shape = (2, 4)
        self.assertTrue(isinstance(m1 / m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 / m2).data) \
                and self.assertEqual(expected_shape, (m1 / m2).shape)

    def test_division_3(self):
        # basic (3x3) with (3x3) division matrix with nested list instance method. The second matrix has all it component negative
        m1 = matrix.Matrix([[2, 4, 5, 6], [11, -12, 13, -14]])
        m2 = 0.5
        expected_val = matrix.Matrix([[4.0, 8.0, 10.0, 12.0], [22.0, -24.0, 26.0, -28.0]])
        expected_shape = (2, 4)
        self.assertTrue(isinstance(m1 / m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 / m2).data) \
                and self.assertEqual(expected_shape, (m1 / m2).shape)

    def test_division_4(self):
        # basic (3x3) with (3x3) division matrix with nested list instance method. The second matrix has all it component negative
        m2 = 10
        m1 = matrix.Matrix([[2]])
        expected_val = matrix.Matrix([[5]])
        expected_shape = (1, 1)
        self.assertTrue(isinstance(m1 / m2, matrix.Matrix)) \
            and self.assertEqual(expected_val, (m1 / m2).data) \
                and self.assertEqual(expected_shape, (m1 / m2).shape)


    # Tests group 11
    def test_error_division_1(self):
        # division between a (2x4) and a (2x2) Matrix
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = 0
        with self.assertRaises(ZeroDivisionError):
            m1 / m2

    def test_error_division_2(self):
        # division beteen a Matrix and a nested list of the same shape
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = [[0, 0, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ArithmeticError):
            m1 / m2

    def test_error_division_3(self):
        # division between (3x3) with tuple
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        m2 = (2, 3)
        with self.assertRaises(ArithmeticError):
            m1 / m2

    # Tests group 12
    def test_transpose_1(self):
        m1 = matrix.Matrix([[1, 2, 3, 10.5], [4, 5, 6, -5.2]])
        expected_val = matrix.Matrix([[1, 4], [2, 5], [3, 6], [10.5, -5.2]])
        expected_shape = (4,2)
        self.assertTrue(isinstance(m1.T(), matrix.Matrix)) \
            and self.assertEqual(expected_val, m1.T().data) \
                and self.assertEqual(expected_shape, m1.T().shape)

    def test_transpose_2(self):
        m1 = matrix.Matrix([[1, 2], [3, 4]])
        expected_val = matrix.Matrix([[1, 3], [2, 4]])
        expected_shape = (2,2)
        self.assertTrue(isinstance(m1.T(), matrix.Matrix)) \
            and self.assertEqual(expected_val, m1.T().data) \
                and self.assertEqual(expected_shape, m1.T().shape)


class TestVectorMethods(TestCase):
    # Tests group 1
    def test_basic_instance_list_1(self):
        expected_val = [[0], [0]]
        expected_shape = (2,1)
        M = matrix.Vector(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_list_2(self):
        expected_val = [[1, 2]]
        expected_shape = (1,2)
        M = matrix.Vector(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_list_3(self):
        expected_val = [[21], [42], [63]]
        expected_shape = (3,1)
        M = matrix.Vector(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_list_4(self):
        expected_val = [[21, 42, 63]]
        expected_shape = (1,3)
        M = matrix.Vector(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    # Tests group 2
    def test_basic_instance_shape_1(self):
        expected_val = [[0], [0]]
        expected_shape = (2,1)
        M = matrix.Vector((2,1))
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_shape_2(self):
        expected_val = [[0], [0], [0]]
        expected_shape = (3,1)
        M = matrix.Vector(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_shape_3(self):
        expected_val = [[0, 0, 0]]
        expected_shape = (1,3)
        M = matrix.Vector(expected_val)
        self.assertTrue(check_data_equality(expected_val, M.data) and check_shape(expected_shape, M.shape))

    def test_basic_instance_shape_4(self):
        expected_shape = (0,6)
        with self.assertRaises(ValueError):
            M = matrix.Vector(expected_shape)

    def test_basic_instance_shape_5(self):
        expected_shape = (-2,6)
        with self.assertRaises(ValueError):
            M = matrix.Vector(expected_shape)

    # Tests group 4
    def test_addition_1(self):
        # basic (2x2) with (2x2) addition matrix with shape instance method
        v1 = matrix.Vector((2, 1))
        v2 = matrix.Vector((2, 1))
        v3 = v1 + v2
        expected_val = [[0], [0]]
        expected_shape = (2, 1)
        self.assertTrue(isinstance(v3, matrix.Vector) \
            and check_data_equality(expected_val, v3.data) \
                and check_shape(expected_shape, v3.shape))

    def test_addition_2(self):
        # basic (3x3) with (3x3) addition matrix with nested list instance method
        v1 = matrix.Vector([[1], [2], [3]])
        v2 = matrix.Vector([[1], [1], [1]])
        v3 = v1 + v2
        expected_val = [[2], [3], [4]]
        expected_shape = (3, 1)
        self.assertTrue(isinstance(v3, matrix.Vector) \
            and check_data_equality(expected_val, v3.data) \
                and check_shape(expected_shape, v3.shape))

    def test_addition_3(self):
        # basic (3x3) with (3x3) addition matrix with nested list instance method. The second matrix has all it component negative
        v1 = matrix.Vector([[1, 2, 3, 10.5]])
        v2 = matrix.Vector([[-1, -1, -1, -1]])
        v3 = v1 + v2
        expected_val = [[0, 1, 2, 9.5]]
        expected_shape = (1, 4)
        self.assertTrue(isinstance(v3, matrix.Matrix)) \
            and self.assertAlmostEqual(expected_val, v3.data) \
                and self.assertEqual(expected_shape, v3.shape)


    # Tests group 5
    def test_error_addition_1(self):
        # Addition between a (2x4) and a (2x2) Matrix
        v1 = matrix.Vector([[1, 2, 3, 10.5]])
        v2 = matrix.Vector([[-1, -1]])
        with self.assertRaises(ArithmeticError):
            v1 + v2

    def test_error_addition_2(self):
        # Addition beteen a Matrix and a nested list of the same shape
        v1 = matrix.Vector([[1], [2], [3], [10.5]])
        v2 = [[0], [0], [0], [0]]
        with self.assertRaises(ArithmeticError):
            v1 + v2

    def test_error_addition_3(self):
        # Addition between (3x3) with tuple
        v1 = matrix.Vector([[1, 2, 3, 10.5]])
        v2 = 3246
        with self.assertRaises(ArithmeticError):
            v1 + v2


    # Tests group 6
    def test_substraction_1(self):
        # basic (2x2) with (2x2) substraction matrix with shape instance method
        v1 = matrix.Vector((2, 1))
        v2 = matrix.Vector((2, 1))
        v3 = v1 - v2
        expected_val = [[0], [0]]
        expected_shape = (2, 1)
        self.assertTrue(isinstance(v3, matrix.Vector) \
            and check_data_equality(expected_val, v3.data) \
                and check_shape(expected_shape, v3.shape))

    def test_substraction_2(self):
        # basic (3x3) with (3x3) substraction matrix with nested list instance method
        v1 = matrix.Vector([[4, 5, 6]])
        v2 = matrix.Vector([[2, 3, 2]])
        v3 = v1 - v2
        expected_val = [[2, 2, 4]]
        expected_shape = (1, 3)
        self.assertTrue(isinstance(v3, matrix.Vector) \
            and check_data_equality(expected_val, v3.data) \
                and check_shape(expected_shape, v3.shape))

    def test_substraction_3(self):
        # basic (3x3) with (3x3) substraction matrix with nested list instance method. The second matrix has all it component negative
        v1 = matrix.Vector([[1], [2], [3], [9.5]])
        v2 = matrix.Vector([[-1], [-1], [-1], [10]])
        v3 = v1 - v2
        expected_val = [[2], [3], [4], [-0.5]]
        expected_shape = (4, 1)
        self.assertTrue(isinstance(v3, matrix.Vector) \
            and check_data_equality(expected_val, v3.data) \
                and check_shape(expected_shape, v3.shape))


    # Tests group 7
    def test_error_substraction_1(self):
        # Substraction between a (2x4) and a (2x2) Matrix
        v1 = matrix.Vector([[1, 2, 3, 10.5]])
        v2 = matrix.Vector([[-1, -1]])
        with self.assertRaises(ArithmeticError):
            v1 + v2

    def test_error_substraction_2(self):
        # Substraction beteen a Matrix and a nested list of the same shape
        v1 = matrix.Vector([[1, 2, 3, 10.5]])
        v2 = [[0, 0, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ArithmeticError):
            v1 + v2

    def test_error_substraction_3(self):
        # Substraction between (3x3) with tuple
        v1 = matrix.Vector([[1, 2, 3, 10.5]])
        v2 = 3246
        with self.assertRaises(ArithmeticError):
            v1 + v2

    # Tests group 8
    def test_multiplication_1(self):
        # basic (2x2) with (2x2) multiplication matrices
        m1 = matrix.Vector([[2], [4]])
        s2 = 5
        expected_val = [[10], [20]]
        expected_shape = (2, 1)
        self.assertTrue(isinstance(m1 * s2, matrix.Vector)) \
            and self.assertEqual(expected_val, (m1 * s2).data) \
                and self.assertEqual(expected_shape, (m1 * s2).shape)

    def test_multiplication_2(self):
        # basic (4x4) with (4x4) multiplication matrices
        s1 = 5
        m2 = matrix.Vector([[2, 4, 6]])
        expected_val = [[10, 20, 30]]
        expected_shape = (1, 3)
        self.assertTrue(isinstance((s1 * m2), matrix.Vector)) \
            and self.assertEqual(expected_val, (s1 * m2).data) \
                and self.assertEqual(expected_shape, (s1 * m2).shape)

    def test_multiplication_vector_matrix(self):
        # basic (3x2) with (2x3) multiplication matrices
        m1 = matrix.Vector([[1, 2]])
        m2 = matrix.Matrix([[3, -7, 2], [0, -3, 5]])
        expected_val = [[3, -13, 12]]
        expected_shape = (1, 3)
        self.assertTrue(isinstance(m1 * m2, matrix.Vector)) \
            and self.assertEqual(expected_val, (m1 * m2).data) \
                and self.assertEqual(expected_shape, (m1 * m2).shape)

    def test_multiplication_vector_matrix_2(self):
        # basic (3x3) multiplication matrix with a float
        m1 = matrix.Matrix([[1, 2, 3], [4, 5, 6]])
        v2 = matrix.Vector([[1], [2], [3]])
        expected_val = [[15], [32]]
        expected_shape = (2, 1)
        self.assertTrue(isinstance(m1 * v2, matrix.Vector)) \
            and self.assertEqual(expected_val, (m1 * v2).data) \
                and self.assertEqual(expected_shape, (m1 * v2).shape)


    # Tests group 9
    def test_error_multiplication_1(self):
        # multiplication between a (2x4) Matrix and a (2x2) Matrix
        m1 = matrix.Vector([[1, 2, 3, 10.5]])
        m2 = matrix.Vector([[-1, -1]])
        with self.assertRaises(AttributeError):
            m1 * m2

    def test_error_multiplication_2(self):
        # multiplication between a (1x3) Matrix and a (2x4) Matrix
        m1 = matrix.Vector([[-1], [-1], [-1]])
        m2 = matrix.Vector([[1], [2], [3], [10.5]])
        with self.assertRaises(AttributeError):
            m1 * m2

    def test_error_multiplication_3(self):
        # multiplication between a (2x4) Matrix and a (2x1) Matrix
        m1 = matrix.Vector([[1, 2, 3, 10.5]])
        m2 = [[1]]
        with self.assertRaises(ArithmeticError):
            m1 * m2

    def test_error_multiplication_4(self):
        # multiplication between a (2x4) Matrix and a (2x1) Matrix
        m1 = matrix.Vector([[1, 2, 3, 10.5]])
        m2 = "Matrix"
        with self.assertRaises(ArithmeticError):
            m1 * m2

    def test_error_multiplication_5(self):
        # multiplication between a (2x4) Matrix and a (2x1) Matrix
        m1 = matrix.Vector([[1], [2], [3], [10.5]])
        m2 = (2,5)
        with self.assertRaises(ArithmeticError):
            m1 * m2

# ---------------------------------------------------------- #
# ________________________   MAIN   ________________________ #
# ---------------------------------------------------------- #
if __name__ == '__main__':
    unittest.main()