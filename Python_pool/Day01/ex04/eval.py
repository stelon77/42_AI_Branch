def checkInput(coefs, words):
    if type(words) is not list:
        return 1
    if len([word for word in words if type(word) is not str]):
        return 1
    if type(coefs) is not list:
        return 1
    if len([coef for coef in coefs if not isinstance(coef, (int, float))]):
        return 1
    if len(words) != len(coefs):
        return 1
    return 0


class Evaluator:
    """
    2 functions to compute the sum of the lengths of every words of a
    given list weighted by a list a coefs."""

    @staticmethod
    def zip_evaluate(coefs, words):
        if checkInput(coefs, words):
            return -1
        total = 0
        for word, coef in zip(words, coefs):
            total += (len(word) * coef)
        return total

    @staticmethod
    def enumerate_evaluate(coefs, words):
        if checkInput(coefs, words):
            return -1
        total = 0
        for i, word in enumerate(words):
            total += (len(word) * coefs[i])
        return total
