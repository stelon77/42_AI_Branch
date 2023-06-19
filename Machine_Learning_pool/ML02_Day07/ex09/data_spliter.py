import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)
    into a training and a test set,
    while respecting the given proportion of examples to be
    kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset
    that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not matrix_and_vector_size_m(x, y):
        return None
    y = reshape_if_needed(y)
    x = reshape_if_needed(x)
    if not isinstance(proportion, float) or proportion < 0. or proportion > 1.:
        return None

    m = x.shape[0]
    nb = int(m * proportion)
    temp = np.concatenate((y, x), axis=1)
    np.random.shuffle(temp)

    train, test = np.split(temp, [nb])
    y_train, x_train = np.split(train, [1], axis=1)
    y_test, x_test = np.split(test, [1], axis=1)

    return x_train, x_test, y_train, y_test

# --------------- TOOLS FUNCTIONS ----------------------------- #


def matrix_and_vector_size_m(x, y):
    if is_numpy_array(x) and \
       is_numpy_array(y) and \
       is_not_empty(x) and \
       is_not_empty(y) and \
       is_made_of_numbers(x) and \
       is_made_of_numbers(y) and \
       is_vertical_vector(y) and \
       is_max_2_dimensions(x) and \
       is_same_number_of_line(x, y):
        return True
    return False


def is_same_number_of_line(x, y):
    return x.shape[0] == y.shape[0]


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def is_vertical_vector(x):
    if x.ndim > 2:
        return False
    if len(x.shape) == 1:
        return True
    if x.shape[1] == 1:
        return True
    return False


def is_max_2_dimensions(x):
    return x.ndim <= 2


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


if __name__ == "__main__":
    x1 = np.array([[1], [42], [300], [10], [59]])
    y = np.array([[0], [1], [0], [1], [0]])
    # Example 0:
    print(data_spliter(x1, y, 0.8))
    # Output:
    """(array([[ 10],[ 42],[ 1],[ 300]]), array([[59]]),
    array([[1],[ 1],[ 0],[ 0]]), array([[0]]))"""
    # Example 1:
    print(data_spliter(x1, y, 0.5))
    # Output:
    """(array([[42],[ 10]]), array([[ 59],[ 300],[ 1]]),
    array([[1],[ 1]]), array([[0],[ 0],[ 0]]))"""
    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]])
    y = np.array([[0], [1], [0], [1], [0]])
    # Example 2:
    print(data_spliter(x2, y, 0.8))
    # Output:
    """(array([[ 10, 42],
    [ 59, 1],
    [ 1, 42],
    [300, 10]]), array([[300, 59]]), array([[0],
    [ 0],[ 0],[ 1]]),array([[1]]))"""
    # Example 3:
    print(data_spliter(x2, y, 0.5))
    # Output:
    """(array([[300, 10],
    [ 1, 42]]),
    array([[ 10, 42],
    [300, 59],
    [ 59, 1]]),
    array([[1],[ 0]]),
    array([[0],[ 1],[ 0]]))"""
