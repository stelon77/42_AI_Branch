import numpy as np

from tools.tools import *


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a vector
    of shape m * 1: (number of training examples, 1).
    y: has to be a numpy.array, a vector
    of shape m * 1: (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of
    iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape 2 * 1.
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of the expected type.
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
    if (not isinstance(alpha, (float, int)) or not isinstance(max_iter, int) or
       max_iter <= 0):
        return None

    X = add_intercept(x)
    new_theta = theta
    for i in range(max_iter):
        nabla_J = np.dot(np.transpose(X), (np.dot(X, new_theta) - y)) / len(x)
        new_theta = new_theta - (nabla_J * alpha)
    return new_theta


if __name__ == "__main__":

    x = np.array([[12.4956442], [21.5007972],
                  [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236],
                  [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([[1], [1]])
    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-6, max_iter=15000)
    print(theta1)
    # Output:
    """
    array([[1.40709365],
    [1.1150909 ]])
    """
    # Example 1:
    print(predict_(x, theta1))
    # Output:
    """
    array([[15.3408728 ],
    [25.38243697],
    [36.59126492],
    [55.95130097],
    [65.53471499]])
    """

    x = np.array(range(1, 101)).reshape(-1, 1)
    y = 0.75*x + 5
    theta = np.array([[1.], [1.]])

    print(fit_(x, y, theta, 1e-5, 2000)) #LINE FROM THE SCALE
    # [[1.01682288]
    # [0.80945473]]

    print(fit_(x, y, theta, 1e-5, 20000)) # WHAT WORKS FOR ME
    # [[4.97090918]
    # [0.75043422]]

    # print(fit_(x, y, theta, 1e-5, 2000))  # LINE FROM THE SCALE
    # [[1.01682288]
    # [0.80945473]]

    # print(fit_(x, y, theta, 1e-4, 500000))  # BEST QUICK ONE
    # [[4.99998198]
    #  [0.75000027]]
