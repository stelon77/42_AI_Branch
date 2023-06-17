#!/usr/bin/env python3
"""
program which determine if a number is even, odd or zero

argument : an int
"""

from sys import argv
import re

if len(argv) != 2:
    if len(argv) > 2:
        print("AssertionError: more than one argument is provided")
else:
    if re.match("[-+]?[0-9]+$", argv[1]) is None:
        print("AssertionError: argument is not integer")
    else:
        nb = int(argv[1])
        if nb == 0:
            print("I’m Zero.")
        elif nb % 2 == 0:
            print("I’m Even.")
        else:
            print("I’m Odd.")
