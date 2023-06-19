import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array
    using the min-max standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if not is_vector_not_empty_numpy(x):
        return None
    x = reshape_if_needed(x)
    min = np.min(x)
    max = np.max(x)
    if min == max:
        print("min = max : no need to normalize")
        return x
    return ((x - min) / (max - min))


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vector(x):
        return True
    return False


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def is_vector(x):
    if x.ndim > 2:
        return False
    if len(x.shape) == 1:
        return True
    if x.shape[1] == 1 or x.shape[0] == 1:
        return True
    return False


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def reshape_if_needed(x):
    if len(x.shape) == 1 or x.shape[0] == 1:
        return x.reshape(len(x), 1)
    return x


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    print(minmax(X))
    # Output:
    """array([0.58333333, 1., 0.33333333, 0.77777778, 0.91666667,
    0.66666667, 0.])"""
    # Example 2:
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    print(minmax(Y))
    # Output:
    """array([0.63636364, 1., 0.18181818, 0.72727273, 0.93939394,
    0.6969697 , 0.])"""
