#!/usr/bin/env python3

"""
programm that print the result of the 4 elementary
mathematical operation of arithmetic and modulp

param : number1 number2
"""
import re
from sys import argv


def usage():
    """
    print the usage of operations.py
    """

    print("Usage: python operations.py <number1> <number2>")
    print("Example:")
    print("\tpython operations.py 10 3")


def operation(nb1, nb2):
    """
    performs the operations and send back five string
    """

    sum = str(nb1 + nb2)
    dif = str(nb1 - nb2)
    product = str(nb1 * nb2)
    if nb2 != 0:
        quotient = str(nb1 / nb2)
        modulo = str(nb1 % nb2)
    else:
        quotient = "ERROR (div by zero)"
        modulo = "ERROR (modulo by zero)"
    return sum, dif, product, quotient, modulo


if len(argv) < 2:
    usage()
elif len(argv) == 2:
    print("InputError: not enough arguments\n")
    usage()
elif len(argv) > 3:
    print("InputError: too many arguments\n")
    usage()
else:
    if (re.match("[-+]?[0-9]+$", argv[1]) is None
       or re.match("[-+]?[0-9]+$", argv[2]) is None):
        print("InputError: only numbers\n")
        usage()
    else:
        a, b, c, d, e = operation(int(argv[1]), int(argv[2]))
        print("Sum:       ", a)
        print("Difference:", b)
        print("Product:   ", c)
        print("Quotient:  ", d)
        print("Remainder: ", e)
