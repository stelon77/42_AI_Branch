from ressources.mylinearregression import *

import numpy as np

##########################################################
#                         DECORATORS                     #
##########################################################


def regularization(fonction):
    """
    decorator used to include the regularisation factor in loss
    """
    def wrap_regularization(*args, **kwargs):
        ret = fonction(*args, **kwargs)
        if ret is None:
            return ret
        y = args[1]
        m = y.shape[0]
        lambda_ = args[0].lambda_

        theta = np.copy(args[0].theta)
        ret += (lambda_ * l2_(theta)) / (2 * m)  # lambda * L2 / 2 * m
        return ret
    return wrap_regularization


def grad_regularization(fonction):
    """
    decorator used to include the regularisation factor in gradient
    """
    def wrap_grad_regularization(*args, **kwargs):
        ret = fonction(*args, **kwargs)
        if ret is None:
            return ret
        x = args[1]
        thetas = np.copy(args[3])
        lambda_ = args[0].lambda_
        m, n = x.shape
        thetas[0] = 0
        ret += (lambda_ * thetas) / m  # lambda * L2
        return ret
    return wrap_grad_regularization


def grad_regularization2(fonction):
    """
    decorator used to include the regularisation factor in gradient_
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
        ret += (lambda_ * thetas) / m  # lambda * L2
        return ret
    return wrap_grad_regularization


def l2_(theta):  # no verification as it has been already checked
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    """
    thetas = np.copy(theta)
    thetas[0, 0] = 0
    return float(np.dot(np.transpose(thetas), thetas))


class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(theta, alpha, max_iter)
        if not isinstance(lambda_, (int, float)):
            return None
        self.lambda_ = lambda_

    def loss_elem_(self, y, y_hat):
        l_e = super().loss_elem_(y, y_hat)
        if l_e is None:
            return None
        return l_e + self.lambda_ * (l2_(self.theta) / y.shape[0])

    @regularization
    def loss_(self, y, y_hat):
        return super().loss_(y, y_hat)

    @grad_regularization
    def gradient(self, x, y, theta):  # used in fit, don't change self.theta
        return super().gradient(x, y, theta)

    @grad_regularization2
    def gradient_(self, x, y):
        return super().gradient_(x, y)

    def fit_(self, x, y):
        return super().fit_(x, y)

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
                print("{} is not a parameter of ridge object".format(key))
        return self

    def l2(self):  # no verification as it has been already checked
        """Computes the L2 regularization of a non-empty numpy.ndarray,
        """
        thetas = np.copy(self.theta)
        thetas[0, 0] = 0
        return float(np.dot(np.transpose(thetas), thetas))


if __name__ == '__main__':
    from ridge import MyRidge
    import sys
    # import sklearn.linear_model.ridge as Ridge

    X = np.arange(1, 10).reshape(-1, 1)
    y = 5 * X - 2

    thetas_1 = np.array([[1], [1]])
    model_1 = MyRidge(thetas_1)
    model_2 = MyRidge(thetas_1)

    print("model_1.thetas = {}".format(model_1.theta))
    print("model_1.alpha = {}".format(model_1.alpha))
    print("model_1.max_iter = {}".format(model_1.max_iter))
    print("model_1.lambda_ = {}".format(model_1.lambda_))
    print(model_1.get_params_())

    # set_params is working properly:
    model_1.set_params_({'max_iter': 10000, 'alpha': 0.01})
    model_2.set_params_({'max_iter': 10000, 'alpha': 0.01, 'lambda_': 0.0})

    # get_params is working properly
    print(model_1.get_params_())
    print(model_2.get_params_())

    # fit_ is working properly
    model_1.fit_(X, y)
    model_2.fit_(X, y)

    # Value of thetas after training
    print("thetas values of model_1:\n{}".format(model_1.theta))
    print("thetas values of model_2:\n{}".format(model_2.theta))
    # Predict is working properly
    pred_1 = model_1.predict_(X)
    pred_2 = model_2.predict_(X)

    # loss_ is working properly
    loss_1 = model_1.loss_(y, pred_1)
    loss_2 = model_2.loss_(y, pred_2)
    print("loss_1 = {}".format(loss_1))
    print("loss_2 = {}".format(loss_2))

    # loss_elem_ is working properly:
    loss_elem_1 = model_1.loss_elem_(y, pred_1)
    loss_elem_2 = model_2.loss_elem_(y, pred_2)
    print("value of loss_elem_1:\n{}".format(loss_elem_1))
    print("value of loss_elem_2:\n{}".format(loss_elem_2))

    print("moyenne des loss_elem_1 / 2 = {}".format(np.mean(loss_elem_1) / 2))
    print("moyenne des loss_elem_2 / 2 = {}".format(np.mean(loss_elem_2) / 2))

    # l2 is working properly:
    l2_1 = model_1.l2()
    l2_2 = model_2.l2()
    print("value of l2 of model_1:\n{}".format(l2_1))
    print("value of l2 of model_2:\n{}".format(l2_2))

    # gradient_ is working properly
    print("gradient of model_1 after fit:\n", model_1.gradient_(X, y))
    print("gradient of model_2 after fit:\n", model_2.gradient_(X, y))

    # Checking the management of the type and value parameters:
    thetas_incorrect1 = '42AI'
    thetas_incorrect2 = np.array([['Intelligence'], ['Artificial'], ['42AI']])
    thetas_incorrect3 = np.array([[1.0], [-1.0], ['42AI']])

    try:
        model = MyRidge(thetas_incorrect1)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    try:
        model = MyRidge(thetas_incorrect2)
        print(model)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    try:
        model = MyRidge(thetas_incorrect3)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    print("max_iter errors")
    max_iter_incorrect1 = np.array([100])
    max_iter_incorrect2 = '42AI'
    max_iter_incorrect3 = 1000.0
    try:
        model = MyRidge(theta=thetas_1, max_iter=max_iter_incorrect1)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
    try:
        model = MyRidge(theta=thetas_1, max_iter=max_iter_incorrect2)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
    try:
        model = MyRidge(theta=thetas_1, max_iter=max_iter_incorrect3)
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    X = np.arange(0, 100).reshape(10, 10)
    y_test1 = np.random.rand(12).reshape(-1, 1)
    y_test2 = np.random.rand(13).reshape(-1, 1)

    # Mismatch between X and thetas. None is expected
    print(model_1.predict_(X))

    # Mismatch between y_test1 and y_test2.
    try:
        print(model_1.loss_(y_test1, y_test2))
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    # Mismatch between X and thetas. None is expected
    try:
        print(model_1.loss_elem_(y_test1, y_test2))
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)

    # Mismatch between X and thetas. None is expected
    try:
        print(model_1.fit_(X, y))
    except SystemExit:
        s = '>> method called sys.exit'
        print(s, file=sys.stderr)
