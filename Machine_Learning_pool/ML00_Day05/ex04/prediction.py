import numpy as np


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
    if not tool.is_vector_not_empty_numpy(x) or \
       not tool.is_vector_not_empty_numpy_2_x_1(theta):
        return None
    x = tool.reshape_if_needed(x)
    theta = tool.reshape_if_needed(theta)

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
    # Example 1:
    theta1 = np.array([[5], [0]])
    print(predict_(x, theta1))
    # Ouput:
    """
    array([[5.],[5.],[5.],[5.],[5.]])
    """
    # Do you remember why y_hat contains only 5’s here?

    # Example 2:
    theta2 = np.array([[0], [1]])
    print(predict_(x, theta2))
    # Output:
    """
    array([[1.],[2.],[3.],[4.],[5.]])
    """
    # Do you remember why y_hat == x here?

    # Example 3:
    theta3 = np.array([[5], [3]])
    print(predict_(x, theta3))
    # Output:
    """
    array([[ 8.],[11.],[14.],[17.],[20.]])
    """

    # Example 4:
    theta4 = np.array([[-3], [1]])
    print(predict_(x, theta4))
    # Output:
    """
    array([[-2.],[-1.],[ 0.],[ 1.],[ 2.]])
    """
