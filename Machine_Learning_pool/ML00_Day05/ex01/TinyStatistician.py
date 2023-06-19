import numpy as np


def check_args(x):
    if not isinstance(x, (list, np.ndarray)):
        return None
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x.flatten().tolist()
    if isinstance(x, list) and not x:
        return None
    for n in x:
        if not isinstance(n, (float, int)):
            return None
    return x


class TinyStatistician:
    def __init__(self):
        pass

    @staticmethod
    def mean(x):
        x = check_args(x)
        if x is None:
            return None

        total = 0
        length = len(x)
        for nb in x:
            total += nb
        return (float(total / length))

    @staticmethod
    def median(x):
        x = check_args(x)
        if x is None:
            return None

        length = len(x)
        x = sorted(x)
        if length % 2 == 1:
            return (float(x[length // 2]))
        else:
            a = x[int(length / 2) - 1]
            b = x[int(length / 2)]
            return (float((a + b) / 2))

    @staticmethod  # pas sur que soit exact
    def quartile(x):
        x = check_args(x)
        if x is None:
            return None

        length = len(x)
        x = sorted(x)
        first = length / 4
        third = length * 3 / 4
        if first == float(length // 4):
            a = x[int(first) - 1]
            b = x[int(third) - 1]
            return [float(a), float(b)]

        else:
            a = (x[int(first)])
            b = (x[int(third)])
            return [float(a), float(b)]

    @staticmethod
    def percentile(x, p):
        x = check_args(x)
        if x is None:
            return None
        if not isinstance(p, (int, float)) or p < 0 or p > 100:
            return None
        length = len(x)
        x = sorted(x)
        percent = length * p / 100
        if percent == float((length * p) // 100):
            a = x[int(percent) - 1]
            return float(a)
        else:
            a = (x[int(percent)])
            return float(a)

    @staticmethod
    def var(x):
        meanX = TinyStatistician.mean(x)
        length = len(x)
        total = 0
        for nb in x:
            total += (nb - meanX)**2
        return (float(total / length))

    @staticmethod
    def std(x):
        return (float(TinyStatistician.var(x)**0.5))
