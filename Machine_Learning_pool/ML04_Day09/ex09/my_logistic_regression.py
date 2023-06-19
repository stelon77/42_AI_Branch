import numpy as np
from utils import *

##########################################################
#                         DECORATORS                     #
##########################################################


def grad_regularization(fonction):
    """
    decorator used to include the regularisation factor L2
    """
    def wrap_grad_regularization(*args, **kwargs):
        ret = fonction(*args, **kwargs)
        if ret is None:
            return ret
        x = args[1]
        thetas = np.copy(args[0].theta)
        lambda_ = args[0].lambda_
        m, n = x.shape
        thetas[0] = 0
        ret += (lambda_ * thetas) / m
        return ret
    return wrap_grad_regularization


def loss_regularization(fonction):
    """
    decorator used to include the regularisation factor L2
    """
    def wrap_loss_regularization(*args, **kwargs):
        ret = fonction(*args, **kwargs)
        if ret is None:
            return ret
        m = args[1].shape[0]
        lambda_ = args[0].lambda_
        theta = args[0].theta
        ret += (lambda_ * l2_(theta) / (2 * m))
        return ret
    return wrap_loss_regularization


def l2_(theta):  # no verification as it has been already checked
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    """
    thetas = np.copy(theta)
    thetas[0, 0] = 0
    return float(np.dot(np.transpose(thetas), thetas))


class MyLogisticRegression():
    supported_penalities = ['l2']  # We consider l2 penality only.
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2',
                 lambda_=1.0):
        if not isinstance(alpha, (float, int)) or \
           not isinstance(lambda_, (float, int)) or \
           not isinstance(max_iter, int) or \
           not isinstance(theta, (np.ndarray, list, tuple)):
            print("bad arguments provided")
            exit(1)
        thetas = np.array(theta)
        if not is_vector_not_empty_numpy(thetas):
            print("bad arguments provided")
            exit(1)
        self.theta = thetas
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalities else 0

    @check_x_theta
    def predict_(self, x):
        """Computes the vector of prediction y_hat
        from two non-empty numpy.ndarray.
        """
        X = self.add_intercept(x)
        return self.sigmoid_(np.dot(X, self.theta))

    @staticmethod
    def sigmoid_(x):  # No protection, it has been checked
        """
        Compute the sigmoid of a vector.
        """
        return (1 / (1 + np.exp(-x)))

    @staticmethod
    def add_intercept(x):  # protection removed as x has been checked
        """Adds a column of 1's to the non-empty numpy.array x.
        """
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    @check_y_yhat
    def loss_elem_(self, y, yhat):
        """
        Description:
        Calculates all the elements of the loss function.
        """
        m = y.shape[0]

        one = np.ones((m, 1))
        epsilon = np.full((m, 1), 1e-15)
        term_1 = np.log(yhat + epsilon)
        term_2 = np.log(one - yhat + epsilon)
        loss_elem = np.multiply(y, term_1) + np.multiply(one - y, term_2)
        loss_elem = - loss_elem  # / m
        reg_term = (l2_(self.theta) * self.lambda_) / (2 * m)  # * m)
        reg = np.full((m, 1), reg_term)
        loss_elem += reg_term
        return loss_elem

    @check_y_yhat
    @loss_regularization
    def loss_(self, y, yhat):
        """
        Computes the logistic loss value.
        """
        m = y.shape[0]
        one = np.ones((m, 1))
        epsilon = np.full((m, 1), 1e-15)
        loss = float(np.dot(np.transpose(y),
                     np.log(yhat + epsilon)) +
                     np.dot(np.transpose(one - y),
                     np.log(one - yhat + epsilon)))
        loss = - loss / m
        return loss

    @check_x_y_theta
    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        """
        # J_history = np.zeros((self.max_iter, 1))  # #####
        m = x.shape[0]
        for i in range(self.max_iter):
            self.theta = self.theta - (self.alpha * self.log_gradient(x, y))
            if np.isinf(self.theta).any() or np.isnan(self.theta).any():
                print("gradient is diverging, choose a smaller alpha")
                return None
            # y_hat = self.predict_(x)  # #######
            # loss = self.loss_(y, y_hat)  # #######
            # J_history[i] = loss  # ########
        # visual(J_history) ###########
        return self

    @grad_regularization
    def log_gradient(self, x, y):  # protections removed to optimize algorithm
        """Computes a gradient vector from three non-empty numpy.ndarray,
        """
        m = x.shape[0]
        y_hat = self.predict_(x)
        X = self.add_intercept(x)
        return np.dot(np.transpose(X), (y_hat - y)) / m

    def get_params_(self):  # returns a dictionnary
        return self.__dict__

    def set_params_(self, params):
        """
        params must be a dictionnary, to change the parameters
        """
        if not isinstance(params, dict):
            print("params should be a dictionnary")
            return self
        for key, value in params.items():
            if key in self.__dict__.keys():
                if not isinstance(value, type(self.__dict__[key])):
                    print("TypeError for variable {}".format(key))
                else:
                    self.__dict__[key] = value
            else:
                print("{} is not a parameter of logistic regression object"
                      .format(key))
        return self


if __name__ == "__main__":
    from my_logistic_regression import MyLogisticRegression as mylogr
    import sys

    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    # Example 1:
    model1 = mylogr(theta, lambda_=5.0)
    print(model1.penalty)
    # Output
    """l2"""
    print(model1.lambda_)
    # Output
    """5.0"""
    # Example 2:
    model2 = mylogr(theta, penalty=None)
    print(model2.penalty)
    # Output
    """None"""
    print(model2.lambda_)
    # Output
    """ 0.0"""
    # Example 3:
    model3 = mylogr(theta, penalty=None, lambda_=2.0)
    print(model3.penalty)
    # Output
    """None"""
    print(model3.lambda_)
    # Output
    """0.0"""

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])

    mylr = mylogr([2, 0.5, 7.1, -4.3, 2.09], alpha=5e-3, max_iter=10000)
    print("mylr.theta = {}".format(mylr.theta))
    print("mylr.alpha = {}".format(mylr.alpha))
    print("mylr.penalty = {}".format(mylr.penalty))
    print("mylr.lambda_ = {}".format(mylr.lambda_))
    print("mylr.max_iter = {}".format(mylr.max_iter))

    print("# Example 0:")
    my_res = mylr.predict_(X)
    print("my prediction:".ljust(25), my_res.reshape(1, -1))

    print("# Example 1:")
    my_res = mylr.loss_(Y, my_res)
    print("my loss:".ljust(25), my_res)

    print("# Example 2:")
    mylr.fit_(X, Y)
    my_res = mylr.theta
    print("my theta after fit:".ljust(25), my_res.reshape(1, -1))

    print("# Example 3:")
    my_res = mylr.predict_(X)
    print("my prediction:".ljust(25), my_res.reshape(1, -1))

    print("# Example 4:")
    my_res2 = mylr.loss_(Y, my_res)
    my_le = mylr.loss_elem_(Y, my_res)
    print("my loss:".ljust(25), my_res2)
    print("mylosselt = ", my_le)
    print(my_le.mean())

    mylr_lb0 = mylogr([2, 0.5, 7.1, -4.3, 2.09],
                      alpha=5e-3, max_iter=10000, penalty=None)
    mylr_lb1 = mylogr([2, 0.5, 7.1, -4.3, 2.09],
                      alpha=5e-3, max_iter=10000, lambda_=0)
    mylr_lb2 = mylogr([2, 0.5, 7.1, -4.3, 2.09],
                      alpha=5e-3, max_iter=10000, lambda_=1)
    mylr_lb3 = mylogr([2, 0.5, 7.1, -4.3, 2.09],
                      alpha=5e-3, max_iter=10000, lambda_=5)
    print(mylr_lb0.get_params_())
    # Checking the management of the type and value parameters:
    thetas_incorrect1 = '42AI'
    thetas_incorrect2 = np.array([['Intelligence'], ['Artificial'], ['42AI']])
    thetas_incorrect3 = np.array([[1.0], [-1.0], ['42AI']])

    try:
        model = mylogr(thetas_incorrect1)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
    try:
        model = mylogr(thetas_incorrect2)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    try:
        model = mylogr(thetas_incorrect3)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    print("incorrect max_iter")
    max_iter_incorrect1 = np.array([100])
    max_iter_incorrect2 = '42AI'
    max_iter_incorrect3 = 1000.0
    thetas_1 = np.array([[1], [1]])

    try:
        model = mylogr(theta=thetas_1, max_iter=max_iter_incorrect1)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
    try:
        model = mylogr(theta=thetas_1, max_iter=max_iter_incorrect2)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
    try:
        model = mylogr(theta=thetas_1, max_iter=max_iter_incorrect3)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    X = np.arange(0, 100).reshape(10, 10)
    y_test1 = np.random.rand(12).reshape(-1, 1)
    y_test2 = np.random.rand(13).reshape(-1, 1)
    # Mismatch between X and thetas. None is expected
    print(mylr_lb0.predict_(X))  # None

    # Mismatch between y_test1 and y_test2.
    try:
        print(mylr_lb0.loss_(y_test1, y_test2))
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    # Mismatch between X and thetas. None is expected
    try:
        print(mylr_lb0.loss_elem_(y_test1, y_test2))
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
    # Mismatch between X and thetas. None is expected
    try:
        print(mylr_lb0.fit_(X, y_test1))
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
