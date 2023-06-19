import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_not_empty_numpy(x) or \
       not is_vector_not_empty_numpy_2_x_1(theta):
        return None
    x = reshape_if_needed(x)
    theta = reshape_if_needed(theta)

    def f(x):
        return (theta[0] + theta[1] * x)

    return(f(x))


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def is_vertical_vector(x):
    if x.ndim > 2:
        return False
    if len(x.shape) == 1 or x.shape[1] == 1:
        return True
    return False


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


def is_vector_not_empty_numpy_2_x_1(x):
    if not is_vector_not_empty_numpy(x):
        return False
    if x.shape not in ((2,), (2, 1)):
        return False
    return True


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


if __name__ == '__main__':
    x = np.arange(1, 6).reshape(-1, 1)
    # Example 1:
    theta1 = np.array([[5], [0]])
    print(simple_predict(x, theta1), '\n')
    # Ouput:
    # array([[5.],[5.],[5.],[5.],[5.]])
    # Do you understand why y_hat contains only 5’s here?

    # Example 2:
    theta2 = np.array([[0], [1]])
    print(simple_predict(x, theta2), '\n')
    # Output:
    # array([[1.],[2.],[3.],[4.],[5.]])
    # Do you understand why y_hat == x here?

    # Example 3:
    theta3 = np.array([[5], [3]])
    print(simple_predict(x, theta3), '\n')
    # Output:
    # array([[ 8.],[11.],[14.],[17.],[20.]])

    # Example 4:
    theta4 = np.array([[-3], [1]])
    print(simple_predict(x, theta4), '\n')
    # Output:
    # array([[-2.],[-1.],[ 0.],[ 1.],[ 2.]])
