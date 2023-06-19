import numpy as np


class bcolors:
    OK = '\033[92m'  # green
    WARNING = '\033[93m'  # yellow
    FAIL = '\033[91m'  # red
    RESET = '\033[0m'  # reset color

##########################################################
#                         DECORATORS                     #
##########################################################


def check_x_y_theta(fonction):
    """
    decorator used to check x and y and theta
    """
    def wrap_x_y_theta(*args, **kwargs):
        if not corresponding_size_matrix(args[1], args[0].theta) or\
           not is_vector_not_empty_numpy(args[2]) or \
           not is_same_number_of_lines(args[1], args[2]):
            return None
        y = reshape_if_needed(args[2])
        args[0].theta = reshape_if_needed(args[0].theta)
        if (not isinstance(args[0].alpha, float) or
           not isinstance(args[0].max_iter, int) or
           args[0].max_iter <= 0 or args[0].alpha <= 0):
            return None
        ret = fonction(args[0], args[1], y)
        return ret
    return wrap_x_y_theta


def check_y_yhat(fonction):
    """
    decorator used to check y and y_hat
    """
    def wrap_y_yhat(*args, **kwargs):
        if not is_vector_not_empty_numpy_same_size(args[2], args[1]):
            return None
        y = reshape_if_needed(args[1])
        y_hat = reshape_if_needed(args[2])
        new_args = [args[0], y, y_hat]
        ret = fonction(*new_args, **kwargs)
        return ret
    return wrap_y_yhat


def check_x_theta(fonction):
    """
    decorator used to check x and theta
    """
    def wrap_x_theta(*args, **kwargs):
        if not corresponding_size_matrix(args[1], args[0].theta):
            return None
        x = reshape_if_needed(args[1])
        args[0].theta = reshape_if_needed(args[0].theta)
        ret = fonction(args[0], x)
        return ret
    return wrap_x_theta


def check_x(fonction):
    """
    decorator used to check x
    """
    def wrap_x(*args, **kwargs):
        if not is_vector_not_empty_numpy_or_scalar(args[0]):
            return None
        x = reshape_if_needed(args[0])
        ret = fonction(x)
        return ret
    return wrap_x

#################################################################
# --------------- TOOLS FUNCTIONS ----------------------------- #
#################################################################


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


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_scalar(x):
    return x.ndim == 0


def is_max_2_dimensions(x):
    return x.ndim <= 2


def is_matrix_size_corresponding(x, theta):
    return ((x.shape[1] + 1) == len(theta))


def is_vector_same_size(x, y):
    return(len(x) == len(y))


def is_same_number_of_lines(x, y):
    return(x.shape[0] == y.shape[0])


def is_vector_not_empty_numpy_or_scalar(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       (is_scalar(x) or
       is_vertical_vector(x)):
        return True
    return False


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


def is_vector_not_empty_numpy_same_size(x, y):
    if is_vector_not_empty_numpy(x) and \
       is_vector_not_empty_numpy(y) and \
       is_vector_same_size(x, y):
        return True
    return False


def is_matrix_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_max_2_dimensions(x) and \
       is_made_of_numbers(x):
        return True
    return False


def corresponding_size_matrix(x, theta):
    if is_matrix_not_empty_numpy(x) and \
       is_vector_not_empty_numpy(theta) and \
       is_matrix_size_corresponding(x, theta):
        return True
    return False
