def adder(a: int, b: int) -> int:
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


def multiplier(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a * b,
    bitwise
    """
    if not(isinstance(a, int)) or not(isinstance(b, int)) or a < 0 or b < 0:
        print("Only natural numbers ")
        return None

    if b > a:
        a, b = b, a  # to be faster
    res = 0
    while b > 0:
        if b & 1:  # si b est impair
            res = adder(res, a)
        a = a << 1
        b = b >> 1
    return res


def main():
    print(multiplier(2, 4))
    print(multiplier(4, 2))
    print(multiplier(21, 2))
    print(multiplier(7, 9))
    print(multiplier(1234567890, 9876543210))


if __name__ == "__main__":
    main()
