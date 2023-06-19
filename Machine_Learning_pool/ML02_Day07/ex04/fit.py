import numpy as np


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of shape m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of shape m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations
    done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape (number of features + 1, 1).
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not corresponding_size_matrix(x, theta):
        return None
    theta = reshape_if_needed(theta)
    if not is_vector_not_empty_numpy(y):
        return None
    y = reshape_if_needed(y)
    if x.shape[0] != y.shape[0]:
        return None
    if not isinstance(alpha, float) or not isinstance(max_iter, int) or \
       alpha <= 0 or max_iter <= 0:
        return None

    new_theta = theta
    for i in range(max_iter):
        new_theta = new_theta - (alpha * gradient(x, y, new_theta))
        if np.isinf(new_theta).any() or np.isnan(new_theta).any():
            print("gradient is diverging, choose a smaller alpha")
            return None
    return new_theta


def gradient(x, y, theta):  # checking has already been made
    """Computes a gradient vector from three
    non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible shapes.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of shapes n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    X = add_intercept(x)
    m = x.shape[0]
    return np.dot(np.transpose(X), (np.dot(X, theta) - y)) / m


def add_intercept(x):
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


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


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
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.],
                 [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)
    # Output:
    """array([[41.99..],[0.97..], [0.77..], [-1.20..]])"""
    # Example 1:
    print(predict_(x, theta2))
    # Output:
    """array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])"""
