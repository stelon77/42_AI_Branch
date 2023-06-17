def gray_code(n: int) -> int:
    """
    A function that takes an integer n and returns its equivalent in Gray code
    """
    if not isinstance(n, int) or n < 0:
        print("Only natural numbers ")
        return None
    return n ^ (n >> 1)


def main():
    for i in range(9):
        print("number {0} is in gray code {1:b},\
 equivalent to {1} in classical binary"
              .format(i, gray_code(i)))


if __name__ == "__main__":
    main()
