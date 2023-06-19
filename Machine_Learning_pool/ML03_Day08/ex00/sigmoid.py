import numpy as np


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


@check_x
def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    return (1 / (1 + np.exp(-x)))

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


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_scalar(x):
    return x.ndim == 0


def is_vector_not_empty_numpy_or_scalar(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       (is_scalar(x) or
       is_vertical_vector(x)):
        return True
    return False


if __name__ == "__main__":
    # Example 1:
    x = np.array([[-4]])
    print(sigmoid_(x))   # Output:
    """array([[0.01798620996209156]])"""
    # Example 2:
    x = np.array([[2]])
    print(sigmoid_(x))    # Output:
    """array([[0.8807970779778823]])"""
    # Example 3:
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))    # Output:
    """array([[0.01798620996209156], [0.8807970779778823], [0.5]])"""
