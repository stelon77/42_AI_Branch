import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will
    return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a
    pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(labels, list)) and (labels is not None):
        return None
    if not isinstance(df_option, bool):
        return None
    if not two_np_same_shape(y, y_hat):
        return None
    if labels is None:
        labels = list(np.unique(np.concatenate((y, y_hat), axis=0)))
    n = len(labels)
    confusion = np.zeros((n, n), dtype=int)

    for i, label_1 in enumerate(labels):
        for j, label_2 in enumerate(labels):
            confusion[i, j] = int(np.sum(((y == label_1) &
                                  (y_hat == label_2)).astype(int)))
    if df_option:
        return pd.DataFrame(data=confusion, index=labels, columns=labels)
    return confusion


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
    from sklearn.metrics import confusion_matrix
    y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                      ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'],
                  ['norminet'], ['dog'], ['norminet']])
    # Example 1:
    # your implementation
    print(confusion_matrix_(y, y_hat))
    # Output:

    """array([[0 0 0]
    [0 2 1]
    [1 0 2]])"""
    # sklearn implementation
    print(confusion_matrix(y, y_hat))
    # Output:
    """array([[0 0 0]
    [0 2 1]
    [1 0 2]])"""
    # Example 2:
    # your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    # Output:
    """array([[2 1]
    [0 2]])"""
    # sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
    # Output:
    """array([[2 1]
    [0 2]])"""
    # Example 3:
    print(confusion_matrix_(y, y_hat, df_option=True))
    # Output:
    """bird dog norminet
    bird 0 0 0
    dog 0 2 1
    norminet 1 0 2"""
    # Example 2:
    print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    # Output:
    """bird dog
    bird 0 0
    dog 0 2"""
