import numpy as np


class MyLinearRegression:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        if not isinstance(alpha, (float, int)) or \
           not isinstance(max_iter, int) or \
           not isinstance(thetas, (np.ndarray, list, tuple)):
            print("bad arguments provided")
            exit(1)
        self.thetas = np.array(thetas)

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a matrix of shape m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of shape m * 1:
        (number of training examples, 1).
        Return:
        new_thetas: numpy.array, a vector of shape (number of features + 1, 1).
        None if there is a matching shape problem.
        None if x, y is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.corresponding_size_matrix(x, self.thetas):
            return None
        self.thetas = MyLinearRegression.reshape_if_needed(self.thetas)
        if not MyLinearRegression.is_vector_not_empty_numpy(y):
            return None
        y = MyLinearRegression.reshape_if_needed(y)
        if x.shape[0] != y.shape[0]:
            return None
        if (not isinstance(self.alpha, float) or
           not isinstance(self.max_iter, int) or
           self.max_iter <= 0 or self.alpha <= 0):
            return None

        X = MyLinearRegression.add_intercept(x)
        new_thetas = self.thetas
        m = x.shape[0]
        for i in range(self.max_iter):
            nabla_J = np.dot(np.transpose(X), (np.dot(X, new_thetas) - y)) / m
            new_thetas = (new_thetas - (self.alpha * nabla_J))
            if np.isinf(new_thetas).any() or np.isnan(new_thetas).any():
                print("gradient is diverging, choose a smaller alpha")
                return None
        self.thetas = new_thetas
        return new_thetas

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of shapes m * n.
        thetas: has to be an numpy.array, a vector of shapes (n + 1) * 1.
        Return:
        y_hat as a numpy.array, a vector of shapes m * 1.
        None if x or thetas are empty numpy.array.
        None if x or thetas shapes are not appropriate.
        None if x or thetas is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.corresponding_size_matrix(x, self.thetas):
            return None
        self.thetas = MyLinearRegression.reshape_if_needed(self.thetas)
        X = MyLinearRegression.add_intercept(x)
        return (np.dot(X, self.thetas))

    def loss_elem_(self, x, y):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        x: has to be an numpy.array, a matrix.
        y: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension
        (number of the training examples,1).
        None if there is a dimension matching problem between y and x.
        None if y or x is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.corresponding_size_matrix(x, self.thetas):
            return None
        self.thetas = MyLinearRegression.reshape_if_needed(self.thetas)
        if not MyLinearRegression.is_vector_not_empty_numpy(y):
            return None
        y = MyLinearRegression.reshape_if_needed(y)
        if x.shape[0] != y.shape[0]:
            return None
        y_hat = MyLinearRegression.predict_(self, x)
        if y_hat is None:
            return None
        dif = y_hat - y
        return(np.square(dif))

    def loss_(self, x, y):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        J_elem = MyLinearRegression.loss_elem_(self, x, y)
        if J_elem is None or len(J_elem) == 0:
            return None
        J_value = np.sum(J_elem) / (2 * len(J_elem))
        return float(J_value)

    def mse_(self, x, y):
        """
        Description:
        Calculates the value of mse loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        return (MyLinearRegression.loss_(self, x, y) * 2)

    @staticmethod
    def gradient(x, y, theta):
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
        X = MyLinearRegression.add_intercept(x)
        m = x.shape[0]
        return np.dot(np.transpose(X), (np.dot(X, theta) - y)) / m

    @staticmethod
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
        if not MyLinearRegression.is_matrix_not_empty_numpy(x):
            return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

# --------------- TOOLS FUNCTIONS ----------------------------- #

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
        if len(x.shape) == 1:
            return True
        if x.shape[1] == 1:
            return True
        return False

    @staticmethod
    def is_max_2_dimensions(x):
        return x.ndim <= 2

    @staticmethod
    def is_vector_same_size(x, y):
        return(len(x) == len(y))

    @staticmethod
    def reshape_vector(x):
        return x.reshape(len(x), 1)

    @staticmethod
    def reshape_if_needed(x):
        if len(x.shape) == 1:
            return x.reshape(len(x), 1)
        return x

    @staticmethod
    def is_matrix_size_corresponding(x, theta):
        return ((x.shape[1] + 1) == len(theta))

    @staticmethod
    def is_made_of_numbers(x):
        return (np.issubdtype(x.dtype, np.floating) or
                np.issubdtype(x.dtype, np.integer))

    @staticmethod
    def is_vector_not_empty_numpy(x):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_vertical_vector(x):
            return True
        return False

    @staticmethod
    def corresponding_size_matrix(x, theta):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_numpy_array(theta) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_not_empty(theta) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_made_of_numbers(theta) and \
           MyLinearRegression.is_vertical_vector(theta) and \
           MyLinearRegression.is_max_2_dimensions(x) and \
           MyLinearRegression.is_matrix_size_corresponding(x, theta):
            return True
        return False

    @staticmethod
    def is_matrix_not_empty_numpy(x):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_max_2_dimensions(x) and \
           MyLinearRegression.is_made_of_numbers(x):
            return True
        return False


if __name__ == '__main__':
    from mylinearregression import MyLinearRegression as MyLR
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLR([[1.], [1.], [1.], [1.], [1]])
    # Example 0:
    print(mylr.predict_(X))
    # Output:
    """array([[8.], [48.], [323.]])"""
    # Example 1:
    print(mylr.loss_elem_(X, Y))
    # Output:
    """array([[225.], [0.], [11025.]])"""
    # Example 2:
    print(mylr.loss_(X, Y))
    # Output:
    """1875.0"""
    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print(mylr.thetas)
    # Output:
    """array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])"""
    # Example 4:
    print(mylr.predict_(X))
    # Output:
    """array([[23.417..], [47.489..], [218.065...]])"""
    # Example 5:
    print(mylr.loss_elem_(X, Y))
    # Output:
    """array([[0.174..], [0.260..], [0.004..]])"""
    # Example 6:
    print(mylr.loss_(X, Y))
    # Output:
    """0.0732.."""
