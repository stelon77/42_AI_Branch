N = 2**16  # ici unsigned entiers 16bits, N doit etre puissance de 2
DIV = 2**32 - 1


def errorMessage():
    """ guess what it does"""
    print("a or b integer between 0 and 65535")
    return None


def errorMessage2():
    """ guess what it does"""
    print("f is a float between 0 and 1")
    return None


def add(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a + b,
    bitwise
    """
    while b != 0:
        carry = a & b  # la retenue est mise qd 1 + 1
        a = a ^ b  # l'addition sans retenue
        b = carry << 1  # decalage vers la gauche de la retenue
    return a


def subtract(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a - b,
    bitwise
    """
    while (b != 0):
        borrow = (~a) & b
        a = a ^ b
        b = borrow << 1
    return a


def mult(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a * b,
    bitwise
    """
    if b > a:
        a, b = b, a  # afin que b soit le plus petit
    res = 0
    while b > 0:
        if b & 1:  # si b est impair
            res = add(res, a)
        a = a << 1
        b = b >> 1
    return res


def div(a, b):
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a / b,
    bitwise
    """
    if b == 0:
        print("Division by zero")
        return None
    res = 0  # the quotient is intialized
    for i in range(31, -1, -1):  # starting our loop
        if b << i <= a:  # checking if b multiplied by 2**i is <= a
            a = subtract(a, b << i)   # subtracting b << i from a
            res = add(res, 1 << i)  # adding 2 power i to the answer
    return res


def rot(n: int, a: int, b: int, ra: int, rb: int) -> tuple:
    """rotate/flip a quadrant appropriately for Hilbert curve"""
    if not rb:
        if ra:
            a = (n - 1) - a
            b = (n - 1) - b
        a, b = b, a
    return a, b


def reverse_map(n: float) -> tuple:
    """
    A function used to decode data from a line into a space
    This is the reverse function of map
    """
    if not isinstance(n, float) or n > 1 or n < 0:
        return errorMessage2()
    ra = rb = a = b = 0
    t = int(n * 4294967295)
    s = 1
    while (s < N):
        ra = 1 & (t >> 1)
        rb = 1 & (t ^ ra)
        a, b = rot(s, a, b, ra, rb)
        a = add(a, mult(s, ra))
        b = add(b, mult(s, rb))
        t = t >> 2
        s = s << 1
    return (a, b)


def reverse_map2(n: float) -> tuple:
    """
    A much simpler function used to decode data from a line into a space
    This is the reverse function of map2
    """
    t = int(n * 4294967295)
    a = t >> 16
    b = t ^ a << 16
    return (a, b)


def map(a: int, b: int) -> float:
    """
    A function to have the coordinate of a 2D point on the Hilbert curve
    """
    if not isinstance(a, int) or not isinstance(b, int):
        return errorMessage()
    if a >> 16 > 0 or b >> 16 > 0 or a < 0 or b < 0:
        return errorMessage()
    ra = rb = d = 0
    s = N >> 1  # 2^16 / 2
    while s > 0:
        ra = (a & s) > 0  # ra = 0 ou 1 pour chaque bit en partant de gauche
        rb = (b & s) > 0  # idem pour b
        # d += s * s * ((3 * ra) ^ rb)
        x = mult(3, ra)
        y = x ^ rb
        z = mult(s, s)
        toAdd = mult(z, y)
        d = add(d, toAdd)
        a, b = rot(s, a, b, ra, rb)
        s = s >> 1
    return d / DIV


def map2(a: int, b: int) -> float:
    """
    A much simpler algorithm of mapping
    """
    if not isinstance(a, int) or not isinstance(b, int):
        return errorMessage()
    if a >> 16 > 0 or b >> 16 > 0 or a < 0 or b < 0:
        return errorMessage()
    new = a << 16 | b
    return new / DIV


def main():
    print(reverse_map(map(8700, 11700)), end="\n\n")
    print(reverse_map(map(87, 117)), end="\n\n")
    print(reverse_map(map(65535, 0)), end="\n\n")
    print(reverse_map(map(0, 65533)), end="\n\n")
    print(reverse_map(map(0, 65532)), end="\n\n")
    print(reverse_map(map(0, 65531)), end="\n\n")
    print(reverse_map(map(65535, 65535)), end="\n\n")
    print(reverse_map(map(0, 0)), end="\n\n")
    a, b = reverse_map(0.333333332867672)
    print(map(a, b), end="\n\n")


if __name__ == "__main__":
    main()
