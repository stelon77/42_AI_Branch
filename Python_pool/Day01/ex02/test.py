from vector import *

victor = Vector([[0.0], [1.0], [2.0], [3.0]])
print(victor.values)
print(victor.shape)
print(victor)
v1 = Vector([[1.0], [1.0], [1.0], [1.0]])
print("----v1------")
print(v1.values)
print(v1.shape)

v2 = Vector(4)
print("----v2------")
print(v2.values)
print(v2.shape)

print("----v3------")
v3 = v1 + v2
print(v3)
print(v3.values)
print(v3.shape)

print("----v4------")
v4 = v1 - v2
print(v4)
print(v4.values)
print(v4.shape)

print("row vector")
v4 = Vector([1.0, 2.5])
v5 = Vector([2.0, 3.0])
print((v4 + v5).values)

print((v4 / 2).values)
# print((2 / v4).values)
print(v4)
print(v4.dot(v5))

print("\n\nles tests du sujet\n")
v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
print(repr(v1 * 5))
print()
v1 = Vector([0.0, 1.0, 2.0, 3.0])
v2 = v1 * 5
print(repr(v2))
print()
print(repr(v1 / 2.0))
print()
print(Vector([[0.0], [1.0], [2.0], [3.0]]).shape)
print()
print(Vector([[0.0], [1.0], [2.0], [3.0]]).values)
print()
print(Vector([0.0, 1.0, 2.0, 3.0]).shape)
print()
print(Vector([0.0, 1.0, 2.0, 3.0]).values)
print()
v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
print(v1.shape)
print()
print(repr(v1.T()))
print()
print(v1.T().shape)
print()
v2 = Vector([0.0, 1.0, 2.0, 3.0])
print("\nVoici v2:")
print(v2)
print(v2.shape)
print()
print("\nvoici sa transformee")
print(v2.T())
print()
print(v2.T().shape)
print("\n\n  fonction DOT")
v1 = Vector(5)
v2 = Vector((3, 8))
print(v1)
print(v2)
print(v1.dot(v2))
