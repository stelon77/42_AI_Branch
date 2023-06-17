from matrix import *

B = Matrix([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])
print(B)
print(B.row_echelon())


B = Matrix([
    [1., 2.],
    [3., 4.]])
print(B)
print(B.row_echelon())

B = Matrix([
    [1., 2.],
    [2., 4.]])
print(B)
print(B.row_echelon())

B = Matrix([
    [8., 5., -2., 4., 28.],
    [4., 2.5, 20., 4., -4.],
    [8., 5., 1., 4., 17.]
])
print(B)
print(B.row_echelon())
