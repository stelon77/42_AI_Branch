import numpy as np
import matplotlib.pyplot as plt


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    tool = tools()
    if not tool.is_two_vector_not_empty_same_size_numpy(x, y):
        return None
    if not tool.is_vector_not_empty_numpy_2_x_1(theta):
        return None
    x = tool.reshape_if_needed(x)
    y = tool.reshape_if_needed(y)
    theta = tool.reshape_if_needed(theta)

    y_hat = predict_(x, theta)
    J = loss_(y, y_hat) * 2  # pour coller a l'exemple

    plt.title("Cost: {:.6f}".format(J))
    plt.scatter(x, y, marker='o')
    plt.plot(x, y_hat, color='orange')
    for i, x_ in enumerate(x):
        plt.plot([x_, x_], [y[i], y_hat[i]], 'r--')
    plt.show()


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array,
    without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    tool = tools()
    if not tool.is_two_vector_not_empty_same_size_numpy(y, y_hat):
        return None
    y = tool.reshape_if_needed(y)
    y_hat = tool.reshape_if_needed(y_hat)
    m = len(y)
    dif = y_hat - y
    J = (np.dot(np.transpose(dif), dif)) / (2 * m)
    return float(J[0][0])


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    tool = tools()
    X = tool.add_intercept(x)
    if X.shape[1] != theta.shape[0]:
        return None
    y_hat = np.dot(X, theta)
    return y_hat


class tools:
    """ All the tools we need """
    def __init__(self):
        pass

    @staticmethod
    def add_intercept(x):
        """Adds a column of 1’s to the non-empty numpy.array x.
        Args:
        x: has to be an numpy.array, a vector of shape m * n.
        Returns:
        x as a numpy.array, a vector of shape m * (n + 1).
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        ones = np.ones((len(x), 1))
        return np.concatenate((ones, x), axis=1)

    @staticmethod
    def is_numpy_array(x):
        return isinstance(x, np.ndarray)

    @staticmethod
    def is_not_empty(x):
        return x.size != 0

    @staticmethod
    def is_vertical_vector(x):
        if x.ndim > 2:
            return False
        if len(x.shape) == 1 or x.shape[1] == 1:
            return True
        return False

    @staticmethod
    def is_made_of_numbers(x):
        return (np.issubdtype(x.dtype, np.floating) or
                np.issubdtype(x.dtype, np.integer))

    @staticmethod
    def is_vector_same_size(x, y):
        return(len(x) == len(y))

    def is_two_vector_not_empty_same_size_numpy(self, x, y):
        if self.is_numpy_array(x) and \
           self.is_numpy_array(y) and \
           self.is_not_empty(x) and \
           self.is_not_empty(y) and \
           self.is_made_of_numbers(x) and \
           self.is_made_of_numbers(y) and \
           self.is_vertical_vector and \
           self.is_vertical_vector(y) and \
           self.is_vector_same_size(x, y):
            return True
        return False

    def is_vector_not_empty_numpy(self, x):
        if self.is_numpy_array(x) and \
           self.is_not_empty(x) and \
           self.is_made_of_numbers(x) and \
           self.is_vertical_vector(x):
            return True
        return False

    def is_vector_not_empty_numpy_2_x_1(self, x):
        if not self.is_vector_not_empty_numpy(x):
            return False
        if x.shape not in ((2,), (2, 1)):
            return False
        return True

    @staticmethod
    def reshape_if_needed(x):
        if len(x.shape) == 1:
            return x.reshape(len(x), 1)
        return x


if __name__ == '__main__':
    x = np.arange(1, 6).reshape(-1, 1)
    y = np.array([[11.52434424], [10.62589482], [13.14755699],
                  [18.60682298], [14.14329568]])
    # Example 1:
    theta1 = np.array([[18], [-1]])
    plot_with_loss(x, y, theta1)

    # Example 2:
    theta2 = np.array([[14], [0]])
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([[12], [0.8]])
    plot_with_loss(x, y, theta3)
