import random


def generator(text, sep=" ", option=None):
    """
    Splits the text according to sep value and yield the substrings.
    option precise if a action is performed to the substrings
    before it is yielded.
    """

    if type(text) is not str:
        print("ERROR")
        exit()
    lst = text.split(sep)
    if option == 'ordered':
        lst.sort()
    elif option == 'unique':
        lst = list(dict.fromkeys(lst))
    elif option == 'shuffle':
        lst = random.sample(lst, len(lst))
    elif option is not None:
        print("ERROR")
        exit()
    for word in lst:
        yield word
