import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_not_empty_numpy(theta):
        return None
    theta = reshape_if_needed(theta)
    thetas = np.copy(theta)
    thetas[0, 0] = 0
    m = thetas.shape[0]
    l2 = 0
    for i in range(m):
        l2 += float(np.power(thetas[i], 2))
    return float(l2)


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_not_empty_numpy(theta):
        return None
    theta = reshape_if_needed(theta)
    thetas = np.copy(theta)
    thetas[0, 0] = 0
    return float(np.dot(np.transpose(thetas), thetas))


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


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


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    print(iterative_l2(x))
    # Output:
    """911.0"""
    # Example 2:
    print(l2(x))
    # Output:
    """911.0"""
    y = np.array([3, 0.5, -6]).reshape((-1, 1))
    # Example 3:
    print(iterative_l2(y))
    # Output:
    """36.25"""
    # Example 4:
    print(l2(y))
    # Output:
    """36.25"""
