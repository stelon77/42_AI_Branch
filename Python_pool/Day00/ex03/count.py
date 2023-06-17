#!/usr/bin/env python3

"""
contains function text_analyzer that displays
the sums of upper-case characters, lower-case characters,
punctuation characters and spaces in a given text.

"""
import string


def text_analyzer(*args):
    """
    This function counts the number of upper characters, lower characters,
    punctuation and spaces in a given text.

    param : text to count
    """

    if len(args) == 0:
        lines = []
        print("What is the text to analyse?")
        lines = input(">> ").split('\n')
        text = ' '.join(lines)
    elif len(args) > 1:
        print("ERROR")
        return
    else:
        text = args[0]

    charNumber = upperNumber = lowerNumber = punctNumber = spaceNumber = 0
    for letter in text:
        charNumber += 1
        if letter in string.punctuation:
            punctNumber += 1
        elif letter.isupper():
            upperNumber += 1
        elif letter.islower():
            lowerNumber += 1
        elif letter.isspace():
            spaceNumber += 1
    print("The text contains", charNumber, "characters:")
    print("-", upperNumber, "upper letters")
    print("-", lowerNumber, "lower letters")
    print("-", punctNumber, "punctuation marks")
    print("-", spaceNumber, "spaces")
