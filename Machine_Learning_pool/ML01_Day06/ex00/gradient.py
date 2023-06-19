import numpy as np
from tools.tools import *


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty
    numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_two_vector_not_empty_same_size_numpy(x, y):
        return None
    x = reshape_if_needed(x)
    y = reshape_if_needed(y)
    if not is_vector_not_empty_numpy_2_x_1(theta):
        return None
    theta = reshape_if_needed(theta)

    y_hat = theta[0] + (x * theta[1])
    gradient = np.empty(shape=(2, 1))
    dif = y_hat - y
    if len(dif) == 0:
        return None
    gradient[0] = np.sum(dif) / len(dif)
    sum = 0
    for a, b in zip(dif, x):
        sum += a * b
    gradient[1] = sum / len(dif)
    return gradient


if __name__ == "__main__":
    # print('c')
    x = np.array([[12.4956442], [21.5007972], [31.5527382],
                 [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287],
                 [46.6793434], [59.5585554]])
    # Example 0:
    theta1 = np.array([[2], [0.7]])
    print(simple_gradient(x, y, theta1))

    theta2 = np.array([[1], [-0.4]])
    print(simple_gradient(x, y, theta2))
