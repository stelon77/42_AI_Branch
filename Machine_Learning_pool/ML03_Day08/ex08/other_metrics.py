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


def is_numpy_same_shape(y, y_hat):
    return y.shape == y_hat.shape


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import (accuracy_score,
                                 precision_score,
                                 recall_score,
                                 f1_score)
    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # Output:
    """0.5"""
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # Output:
    """0.5"""
    # Precision
    # your implementation
    print(precision_score_(y, y_hat))
    # Output:
    """0.4"""
    # sklearn implementation
    print(precision_score(y, y_hat))
    # Output:
    """0.4"""
    # Recall
    # your implementation
    print(recall_score_(y, y_hat))
    # Output:
    """0.6666666666666666"""
    # sklearn implementation
    print(recall_score(y, y_hat))
    # Output:
    """0.6666666666666666"""
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat))
    # Output:
    """0.5"""
    # sklearn implementation
    print(f1_score(y, y_hat))
    # Output:
    """0.5"""
    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                      'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet',
                  'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # Output:
    """0.625"""
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # Output:
    """0.625"""
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    # Output:
    """0.6"""
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))
    # Output:
    """0.6"""
    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    # Output:
    """0.75"""
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog'))
    # Output:
    """0.75"""
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    # Output:
    """0.6666666666666665"""
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog'))
    # Output:
    """0.6666666666666665"""

    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                      'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                  'dog', 'norminet', 'dog', 'norminet'])
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    # Output:
    """0.6666666666666666"""
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet'))
    # Output:
    """0.6666666666666666"""
    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    # Output:
    """0.5"""
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet'))
    # Output:
    """0.5"""
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    # Output:
    """0.5714285714285715"""
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet'))
    # Output:
    """0.5714285714285715"""
