from matrix import *


# creating matrix
ma = Matrix([[1.0, 2.0], [3.0, 4.0], [5, 6]])
mi = Matrix((4, 3))
print("\n-------------------creating matrix------------------------\n")
print("ma shape : ", ma.size())
print("ma  datas : ", ma.data)
print("mi shape : ", mi.shape)
print("mi  datas : ", mi.data)
try:
    print("\nwrong matrix")
    m_false_1 = Matrix((4, 3, 2))
except ValueError as e:
    print(e)

try:
    print("wrong matrix")
    m_false_2 = Matrix([[1.0, 2.0], ["yo", "ya"], [5, 6]])
except TypeError as e:
    print(e)

try:
    print("wrong matrix")
    m_false_3 = Matrix([[1.0, 2.0], [3.0, 4.0], [5, 6, 7]])
except TypeError as e:
    print(e)

print("\n--------------matrix addition-----------------------\n")
mb = Matrix([[2.0, 1.0], [4.0, 3.0], [6, 5]])
print("ma data :", ma.data)
print("mb data: ", mb.data)
mc = ma + mb
# print(isinstance(mc, Matrix))
# # print(mc.shape)
print("mc = ma + mb ", mc.data)
print("mc2 = mb.add(ma)", mb.add(ma))
try:
    print("\nwrong matrix size")
    m_b_bis = Matrix([[1.0, 2.0, 3], [3.0, 4.0, 5], [5, 6, 7]])
    print("mb bis data :", m_b_bis.data)
    mc_bis = ma + m_b_bis
except ArithmeticError as e:
    print(e)

print("\n--------------matrix substraction-----------------------\n")

md = ma - mb
print(repr(md))
# # print(md.shape)
print("md = ma - mb : ", md.data)
try:
    print("\nwrong matrix size")
    m_b_bis = Matrix([[1.0, 2.0, 3], [3.0, 4.0, 5], [5, 6, 7]])
    print("mb bis data :", m_b_bis.data)
    mc_bis = ma - m_b_bis
except ArithmeticError as e:
    print(e)

print("\n--------------matrix multiplication by scalar---------------------\n")
mf = md * 3
print("mf  = md * 3: ", mf.data)
mg = 3 * mf
print("mg  = 3 * mf: ", mg.data)
try:
    print("\nmultiplication by string")
    mg_bis = mf * "3"
except ArithmeticError as e:
    print(e)

print("\n--------------vector creation-----------------------\n")
try:
    v1 = Vector([[1, 2, 3]])  # create a row vector
    print(v1)
    v2 = Vector([[1], [2], [3]])  # create a column vector
    print(v2)
    v2b = Vector([4, 5, 6])
    print(v2b)
    v3 = Vector([[1, 2], [3, 4]])  # return an error
    print(v3)
except ValueError as e:
    print(e)

print("\n--------------vector addition-----------------------\n")
v4 = v1 + v1
print(v4)
v5 = v4 + v2
print(v5)
v6 = v2 + v4
print(v6)
print(v2.add(v4))

print("\n--------------vector subtraction-----------------------\n")
print(v6 - v4)
print(v6.sub(v4))

print("\n--------------vector scaling-----------------------\n")
print(v6.scl(5))
print(v6 * 5)
print(5 * v6)
try:
    v6.scl("a")
except ValueError as e:
    print(e)

print("\n--------------subject examples-----------------------\n")
u = Vector([2, 3])
v = Vector([5, 7])
u = u.add(v)
print(u)
# [7.0]
# [10.0]

u = Vector([2, 3])
v = Vector([5, 7])
u = u.sub(v)
print(u)
# [-3.0]
# [-4.0]

u = Vector([2, 3])
u = u.scl(2)
print(u)
# [4.0]
# [6.0]

u = Matrix([[1, 2], [3, 4]])
v = Matrix([[7, 4], [-2, 2]])
u = u.add(v)
print(u)
# [8.0, 6.0]
# [1.0, 6.0]

u = Matrix([[1, 2], [3, 4]])
v = Matrix([[7, 4], [-2, 2]])
u = u.sub(v)
print(u)
# [-6.0, -2.0]
# [5.0, 2.0]

u = Matrix([[1, 2], [3, 4]])
u = u.scl(2)
print(u)
# [2.0, 4.0]
# [6.0, 8.0]
