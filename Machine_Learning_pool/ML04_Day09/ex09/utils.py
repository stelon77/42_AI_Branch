import numpy as np
import matplotlib.pyplot as plt


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


#################################################################
# --------------- TOOLS FUNCTIONS ----------------------------- #
#################################################################


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)
    into a training and a test set,
    while respecting the given proportion of examples to be
    kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset
    that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not matrix_and_vector_size_m(x, y):
        return None
    y = reshape_if_needed(y)
    x = reshape_if_needed(x)
    if not isinstance(proportion, float) or proportion < 0. or proportion > 1.:
        return None

    m = x.shape[0]
    nb = int(m * proportion)
    temp = np.concatenate((y, x), axis=1)
    np.random.seed(42)  # pour avoir toujours la meme serie
    np.random.shuffle(temp)

    train, test = np.split(temp, [nb])
    y_train, x_train = np.split(train, [1], axis=1)
    y_test, x_test = np.split(test, [1], axis=1)

    return x_train, x_test, y_train, y_test


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising
    its columns to every power in the range of 1 up to the power giveArgs:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    power: has to be an int, the power up to which the columns
    of matrix x are going to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray,
    of shape m * (np), containg the polynomial feature
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not is_matrix_not_empty_numpy(x):
        return None
    x = reshape_if_needed(x)
    if not isinstance(power, int) or power < 1:
        return None
    m, n = x.shape
    return_matrix = np.zeros((m, n * power), dtype=x.dtype)
    for pow in range(power):
        for col in range(n):
            return_matrix[:, (pow * n) + col] = np.power(x[:, col], pow + 1)
    return return_matrix


def zscore_normalization(x):
    if not is_matrix_not_empty_numpy(x):
        return None
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    X = np.divide(np.subtract(x, mu), std)
    return X, mu, std


def normalize_others(x, mu, std):
    X = np.divide(np.subtract(x, mu), std)
    return X


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not two_np_same_shape(y, y_hat):
        return None
    return np.mean((y_hat == y).astype(float))


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on
                which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not two_np_same_shape(y, y_hat):
        return None
    if not isinstance(pos_label, (int, str)):
        return None
    tp = np.sum((y_hat == pos_label) & (y == pos_label)).astype(int)
    fp = np.sum((y_hat == pos_label) & (y != pos_label)).astype(int)
    if tp + fp == 0:
        return None
    return float(tp / (tp + fp))


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on
                which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not two_np_same_shape(y, y_hat):
        return None
    if not isinstance(pos_label, (int, str)):
        return None
    tp = np.sum((y_hat == pos_label) & (y == pos_label)).astype(int)
    fn = np.sum((y_hat != pos_label) & (y == pos_label)).astype(int)
    if fn + tp == 0:
        return None
    return float(tp / (tp + fn))


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which
                to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    p = precision_score_(y, y_hat, pos_label=pos_label)
    r = recall_score_(y, y_hat, pos_label=pos_label)
    if p is None or r is None or (p + r == 0):
        return None
    return float((2 * p * r) / (p + r))


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


def matrix_and_vector_size_m(x, y):
    if is_matrix_not_empty_numpy(x) and \
       is_vector_not_empty_numpy(y) and \
       is_same_number_of_lines(x, y):
        return True
    return False


def two_np_same_shape(y, y_hat):
    if is_numpy_array(y) and \
       is_numpy_array(y_hat) and \
       is_not_empty(y) and \
       is_not_empty(y_hat) and \
       is_numpy_same_shape(y, y_hat):
        return True
    return False


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


def is_numpy_same_shape(y, y_hat):
    return y.shape == y_hat.shape


def visual(vec):
    m = vec.shape[0]
    x = np.array(range(m)).reshape((-1, 1))
    plt.plot(x, vec)
    plt.show()


def sigmoid_predict_all(x):
    return (1 / (1 + np.exp(-x)))
