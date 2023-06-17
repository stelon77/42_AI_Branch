def adder(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a + b,
    bitwise
    """
    if not(isinstance(a, int)) or not(isinstance(b, int)) or a < 0 or b < 0:
        print("Only natural numbers ")
        return None

    while b != 0:
        carry = a & b  # la retenue est mise qd 1 + 1
        a = a ^ b  # l'addition sans retenue
        b = carry << 1  # decalage vers la gauche de la retenue
    return a


def main():
    print(adder(-1, 2))
    print(adder(154, 12))
    print(adder(123, 3))
    print(adder(20000000000, 40000000000))


if __name__ == "__main__":
    main()
