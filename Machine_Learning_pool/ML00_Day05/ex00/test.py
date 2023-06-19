from matrix import *


# creating matrix
ma = Matrix([[1.0, 2.0], [3.0, 4.0], [5, 6]])
mi = Matrix((4, 3))
print("\n-------------------creating matrix------------------------\n")
print("ma shape : ", ma.shape)
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
try:
    print("\nwrong matrix size")
    m_b_bis = Matrix([[1.0, 2.0, 3], [3.0, 4.0, 5], [5, 6, 7]])
    print("mb bis data :", m_b_bis.data)
    mc_bis = ma + m_b_bis
except ArithmeticError as e:
    print(e)

print("\n--------------matrix substraction-----------------------\n")

md = ma - mb
# # print(repr(md))
# # print(md.shape)
print("md = ma - mb : ", md.data)
try:
    print("\nwrong matrix size")
    m_b_bis = Matrix([[1.0, 2.0, 3], [3.0, 4.0, 5], [5, 6, 7]])
    print("mb bis data :", m_b_bis.data)
    mc_bis = ma - m_b_bis
except ArithmeticError as e:
    print(e)

print("\n--------------matrix division by scalar-----------------------\n")
# # print(md)
me = md / 3.0
print("me = md / 3 :", me.data)
try:
    print("\nwrong input : 3 / me")
    mc_bis = 3 / me
except ValueError as e:
    print(e)

try:
    print("\ndivision by 0:")
    mc_bis = md / 0
except ZeroDivisionError as e:
    print(e)

print("\n--------------matrix multiplication by scalar---------------------\n")
mf = me * 3
print("mf  = me * 3: ", mf.data)
mg = 3 * mf
print("mg  = 3 * mf: ", mg.data)
try:
    print("\nmultiplication by string")
    mg_bis = mf * "3"
except ArithmeticError as e:
    print(e)

print("\n--------------matrix multiplication by matrix---------------------\n")


mh = Matrix([[2, 8, 3], [5, 4, 1]])
mi = Matrix([[4, 1], [6, 3], [2, 4]])
mi_bis = Matrix([[4, 1, 3], [6, 3, 1], [2, 4, 3]])
print("mh = ", mh.data)
print("mi = ", mi.data)
print("mi_bis = ", mi_bis.data)
mj = mh * mi
print("mj = mh * mi", mj.data)
mk = mi * mh
print("mk = mi * mh", mk.data)
try:
    print("\nerror of size")
    mk_bis = mi_bis * mh
except AttributeError as e:
    print(e)


print("\n--------------matrix transposition-----------------------\n")
print("mi = ", mi)
ml = mi.T()
print("sa transposee :", ml)

print("mi_bis = ", mi_bis)
mm = mi_bis.T()
print("sa transposee :", mm)

print("\n--------------vector creation-----------------------\n")
try:
    v1 = Vector([[1, 2, 3]])  # create a row vector
    print(v1)
    v2 = Vector([[1], [2], [3]])  # create a column vector
    print(v2)
    v3 = Vector([[1, 2], [3, 4]])  # return an error
    print(v3)
except ValueError as e:
    print(e)

print("\n--------------vector dot-----------------------\n")
try:
    print("v1 dot(v2)")
    v1.dot(v2)
except ValueError as e:
    print(e)
print("v1.dot(v2.T()) :")
v4 = v1.dot(v2.T())
print(v4)
