from matrix import *

u = Matrix([
    [1., -1.],
    [-1., 1.]
])
print(u.determinant())
0.0
u = Matrix([
    [2., 0., 0.],
    [0., 2., 0.],
    [0., 0., 2.]
])
print(u.determinant())
# 8.0
u = Matrix([
    [8., 5., -2.],
    [4., 7., 20.],
    [7., 6., 1.]
])
print(u.determinant())
# -174.0

u = Matrix([
    [1., -2., 4.],
    [3., 5., -1.],
    [2., -6., -3.]
])
print(u.determinant())
# -147.0
u = Matrix([
    [8., 5., -2., 4.],
    [4., 2.5, 20., 4.],
    [8., 5., 1., 4.],
    [28., -4., 17., 1.]
])
print(u.determinant())
# 1032
u = Matrix([
    [2., 0., 1., -3.],
    [-1., 4, -7., 2.],
    [0., 3., 5., 0.],
    [-2., 1., 0., 6.]
])
print(u.determinant())
# 235
