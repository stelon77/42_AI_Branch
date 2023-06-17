import numpy as np
import copy


class ColorFilter:
    def __init__(self):
        pass

    @staticmethod
    def checkArray(array):
        if not (isinstance(array, np.ndarray)):
            return False
        if "float" in str(array.dtype) or "int" in str(array.dtype):
            return True
        return False

    @staticmethod
    def invert(array):
        """
        Inverts the color of the image received as a numpy array.
        Args:
        array: numpy.ndarray corresponding to the image.
        Return:
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if not ColorFilter.checkArray(array):
            return None
        invertArr = 1 - array
        invertArr[:, :, 3:] = array[:, :, 3:]  # transparency
        return invertArr

    @staticmethod
    def to_blue(array):
        """
        Applies a blue filter to the image received as a numpy array.
        Args:
        array: numpy.ndarray corresponding to the image.
        Return:
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if not ColorFilter.checkArray(array):
            return None
        height = array.shape[0]
        width = array.shape[1]
        rgArray = np.zeros((height, width, 2), dtype=array.dtype)
        newArray = np.dstack((rgArray, array[:, :, 2:]))
        return newArray

    @staticmethod
    def to_green(array):
        """
        Applies a green filter to the image received as a numpy array.
        Args:
        array: numpy.ndarray corresponding to the image.
        Return:
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if not ColorFilter.checkArray(array):
            return None
        gArray = copy.deepcopy(array)
        gArray[:, :, :1] = copy.deepcopy(array)[:, :, :1] * 0
        gArray[:, :, 2:3] = copy.deepcopy(array)[:, :, 2:3] * 0
        return gArray

    @staticmethod
    def to_red(array):
        """
        Applies a red filter to the image received as a numpy array.
        Args:
        array: numpy.ndarray corresponding to the image.
        Return:
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if not ColorFilter.checkArray(array):
            return None
        newArray = array - ColorFilter.to_blue(array)
        newArray = newArray - ColorFilter.to_green(array)
        newArray[:, :, 3:] = array[:, :, 3:]
        return newArray

    @staticmethod
    def to_celluloid(array):
        """
        Applies a celluloid filter to the image received as a numpy array.
        Celluloid filter must display at least four thresholds of shades.
        Be careful! You are not asked to apply black contour on the object,
        you only have to work on the shades of your images.
        Remarks:
        celluloid filter is also known as cel-shading or toon-shading.
        Args:
        array: numpy.ndarray corresponding to the image.
        Return:
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if not ColorFilter.checkArray(array):
            return None
        valScale = np.linspace(1, 0, num=10)
        arr = array * 1
        for elt1 in arr:
            for elt2 in elt1:
                for nb in (0, 1, 2):
                    for val in valScale:
                        if elt2[nb] >= val:
                            elt2[nb] = val
                            break
        return arr

    @staticmethod
    def to_grayscale(array, filter, **kwargs):
        """
        Applies a grayscale filter to the image received as a numpy array.
        For filter = 'mean'/'m': performs the mean of RBG channels.
        For filter = 'weight'/'w': performs a weighted mean of RBG channels.
        Args:
        array: numpy.ndarray corresponding to the image.
        filter: string with accepted values in ['m','mean','w','weight']
        weights: [kwargs] list of 3 floats where the sum equals to 1,
        corresponding to the weights of each RBG channels.
        Return:
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if not ColorFilter.checkArray(array):
            return None
        if len(kwargs) == 0:
            if filter not in ("m", "mean"):
                return None
            tmp = np.sum(array[:, :, :3], axis=2)
            tmp = tmp / 3
            grayArray = np.dstack((tmp, tmp, tmp, array[:, :, 3:]))
            return grayArray

        elif len(kwargs) == 1:
            if filter not in ("w", "weight"):
                return None
            for weights in kwargs.values():
                if not isinstance(weights, list) or len(weights) != 3:
                    return None
                argSum = 0
                for nb in weights:
                    if not isinstance(nb, (int, float)):
                        return None
                    argSum += nb
                if argSum != 1:
                    return None
                tmp1 = array[:, :, :1] * weights[0]
                tmp2 = array[:, :, 1:2] * weights[1]
                tmp3 = array[:, :, 2:3] * weights[2]
                tmp4 = np.dstack((tmp1, tmp2, tmp3))
                tmp = np.sum(tmp4, axis=2)
                grayArray = np.dstack((tmp, tmp, tmp, array[:, :, 3:]))
                return grayArray
        else:
            return None


if __name__ == "__main__":
    from srcs.ImageProcessor import ImageProcessor

    cf = ColorFilter()
    imp = ImageProcessor()
    # arr = imp.load("../ressources/elon_canaGAN.png")
    # arr = imp.load("../ressources/ball.png")
    arr = imp.load("../ressources/42AI.png")
    # arr = imp.load("../ressources/477-4773730.png")

    imp.display(cf.invert(arr))
    imp.display(cf.to_blue(arr))
    imp.display(cf.to_green(arr))
    imp.display(cf.to_red(arr))
    imp.display(cf.to_grayscale(arr, "m"))
    imp.display(cf.to_grayscale(arr, 'weight', w=[0.2, 0.3, 0.5]))
    imp.display(cf.to_celluloid(arr))
