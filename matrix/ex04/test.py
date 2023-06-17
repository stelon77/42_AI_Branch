from matrix import *

u = Vector([0., 0., 0.])
print(u.norm_1(), u.norm(), u.norm_inf())
# 0.0, 0.0, 0.0
u = Vector([1., 2., 3.])
print(u.norm_1(), u.norm(), u.norm_inf())
# 6.0, 3.74165738, 3.0
u = Vector([-1., -2.])
print(u.norm_1(), u.norm(), u.norm_inf())
# 3.0, 2.236067977, 2.0
