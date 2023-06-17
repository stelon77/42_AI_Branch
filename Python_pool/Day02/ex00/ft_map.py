def ft_map(function_to_apply, iterable):
    """Map the function to all elements of the iterable.
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
            yield(function_to_apply(elt))
    except TypeError:
        print("ERROR: iterable is not iterable \
or type in iterable has to be compatible with function")
        return None


if __name__ == "__main__":
    # Example 1:
    x = [1, 2, 3, 4, 5]
    y = ["lol", 4, 3, 5, True, 2, ]
    # x= 1
    print(ft_map(lambda dum: dum + 1, x))
    print(list(ft_map(lambda t: t + 1, x)))
    print(list(ft_map(lambda x: x + 2, [])))
    print(list(ft_map(lambda x: x + 2, [1])))
    print(list(ft_map(lambda x: x ** 2, [1, 2, 3, 4, 5])))
    print(list(ft_map(lambda t: t + 1, y)))
