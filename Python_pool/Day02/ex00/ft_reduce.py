def ft_reduce(function_to_apply, iterable):
    """Apply function of two arguments cumulatively.
    Args:
    function_to_apply: a function taking an iterable.
    iterable: an iterable object (list, tuple, iterator).
    Returns:
    A value, of same type of elements in the iterable parameter.
    None if the iterable can not be used by the function.
    """
    if not callable(function_to_apply):
        print("ERROR: function is not callable")
        return None
    try:
        iter(iterable)
    except TypeError:
        print("ERROR: iterable is not iterable")
        return None
    if len(iterable) == 0:
        print("ERROR: iterable needs at least a value")
        return None
    try:
        new = iterable[0]
        for elt in iterable[1:]:
            new = function_to_apply(new, elt)
    except TypeError:
        print("ERROR: iterable can not be used by the function")
        return None
    return new


if __name__ == "__main__":
    # Example 1:
    lst = ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    print(ft_reduce(lambda u, v: u + v, lst))
    print(ft_reduce((lambda x, y: x + y), [1]))
    print(ft_reduce((lambda x, y: x * y), [1, 2, 3, 4]))
    print(ft_reduce((lambda x, y: x + y), [11.2, 3, 5, True, 2, ]))
