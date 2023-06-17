def ft_filter(function_to_apply, iterable):
    """Filter the result of function apply to all elements of the iterable.
    Args:
    function_to_apply: a function taking an iterable.
    iterable: an iterable object (list, tuple, iterator).
    Returns:
    An iterable.
    None if the iterable can not be used by the function.
    """
    if not callable(function_to_apply):
        print("ERROR: function is not callable")
        return None
    try:
        iter(iterable)
        for elt in iterable:
            if function_to_apply(elt):
                yield elt
    except TypeError:
        print("ERROR: iterable is not iterable \
or type in iterable has to be compatible with function")
        return None


if __name__ == "__main__":
    # Example 1:
    x = [1, 2, 3, 4, 5]
    y = ['lol', 4, 3, 5, True, 2, ]
    print(ft_filter(lambda dum: not (dum % 2), x))
    print(list(ft_filter(lambda dum: not (dum % 2), x)))
    print(list(ft_filter(lambda x: x <= 1, [])))
    print(list(ft_filter(lambda x: x < 4, y)))
