import numpy as np


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shapes m * n.
    theta: has to be an numpy.array, a vector of shapes (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of shapes m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not corresponding_size_matrix(x, theta):
        return None
    theta = reshape_if_needed(theta)
    X = add_intercept(x)
    return (np.dot(X, theta))


def add_intercept(x):  # les checks ont ete pratiques avant
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be an numpy.array, a vector of shape m * n.
    Returns:
    x as a numpy.array, a vector of shape m * (n + 1).
    None if x is not a numpy.array.
    None if x is a empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    ones = np.ones((x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)

# --------------- TOOLS FUNCTIONS ----------------------------- #


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


def is_vector_same_size(x, y):
    return(len(x) == len(y))


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_matrix_size_corresponding(x, theta):
    return ((x.shape[1] + 1) == len(theta))


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def corresponding_size_matrix(x, theta):
    if is_numpy_array(x) and \
       is_numpy_array(theta) and \
       is_not_empty(x) and \
       is_not_empty(theta) and \
       is_made_of_numbers(x) and \
       is_made_of_numbers(theta) and \
       is_vertical_vector(theta) and \
       is_matrix_size_corresponding(x, theta):
        return True
    return False


if __name__ == '__main__':
    x = np.arange(1, 13).reshape((4, 3))
    # Example 0:
    theta1 = np.array([[5], [0], [0], [0]])
    print(predict_(x, theta1))
    # Ouput:
    """array([[5.],[ 5.],[ 5.],[ 5.]])"""
    # Do you understand why y_hat contains only 5’s here?
    # Example 1:
    theta2 = np.array([[0], [1], [0], [0]])
    print(predict_(x, theta2))
    # Output:
    """array([[ 1.],[ 4.],[ 7.],[ 10.]])"""
    # Do you understand why y_hat == x[:,0] here?
    # Example 2:
    theta3 = np.array([[-1.5], [0.6], [2.3], [1.98]])
    print(predict_(x, theta3))
    # Output:
    """array([[ 9.64],[ 24.28],[ 38.92],[ 53.56]])"""
    # Example 3:
    theta4 = np.array([[-3], [1], [2], [3.5]])
    print(predict_(x, theta4))
    # Output:
    """array([[12.5],[ 32. ],[ 51.5],[ 71. ]])"""
