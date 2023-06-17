import numpy as np


class ScrapBooker:
    def __init__(self):
        pass

    def errorDatas(self, datas, typeOfData, typeOfSubData=None):
        """
        Return True if the datas are the wrong type
        """
        if not isinstance(datas, typeOfData):
            return True
        if typeOfSubData is not None:
            if len(datas) != 2:
                return True
            for sub in datas:
                if not isinstance(sub, typeOfSubData):
                    return True
        return False

    def crop(self, array, dim, position=(0, 0)):
        """
        Crops the image as a rectangle via dim arguments (being the new
        height and width of the image) from the coordinates given
        by position arguments.
        Args:
        array: numpy.ndarray
        dim: tuple of 2 integers.
        position: tuple of 2 integers.
        Returns:
        new_arr: the cropped numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if self.errorDatas(array, np.ndarray) or \
           self.errorDatas(dim, tuple, int) or \
           self.errorDatas(position, tuple, int):
            return None
        hight = array.shape[0]
        width = array.shape[1]
        if position[0] < 0 or position[0] >= hight or dim[0] <= 0 \
           or position[0] + dim[0] >= hight:
            return None
        if position[1] < 0 or position[1] >= width or \
           dim[1] <= 0 or position[1] + dim[1] >= width:
            return None
        newArray = array[position[0]:position[0] + dim[0],
                         position[1]:position[1] + dim[1]]
        return newArray

    def thin(self, array, n, axis):
        """
        Deletes every n-th line pixels along
        the specified axis (0: vertical, 1: horizontal)
        Args:
        array: numpy.ndarray.
        n: non null positive integer lower than the number of
        row/column of the array (depending of axis value).
        axis: positive non null integer.
        Returns:
        new_arr: thined numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if self.errorDatas(array, np.ndarray) or self.errorDatas(n, int) or \
           self.errorDatas(axis, int):
            return None
        if axis == 0:
            axis = 1
        else:
            axis = 0
        if axis < 0 or axis > 1 or n <= 0:
            return None
        if (axis == 0 and n > array.shape[0]) or \
           (axis == 1 and n > array.shape[1]):
            return None
        newArray = np.delete(array, np.s_[n - 1::n], axis)
        return newArray

    def juxtapose(self, array, n, axis):
        """
        Juxtaposes n copies of the image along the specified axis.
        Args:
        array: numpy.ndarray.
        n: positive non null integer.
        axis: integer of value 0 or 1.
        Returns:
        new_arr: juxtaposed numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if self.errorDatas(array, np.ndarray) or self.errorDatas(n, int) or \
           n <= 0 or (axis != 0 and axis != 1):
            return None
        i = 1
        newArray = array.copy()
        while i < n:
            newArray = np.concatenate((newArray, array), axis)
            i += 1
        return newArray

    def mosaic(self, array, dim):
        """
        Makes a grid with multiple copies of the array. The dim argument
        specifies the number of repetition along each dimensions.
        Args:
        array: numpy.ndarray.
        dim: tuple of 2 integers.
        Returns:
        new_arr: mosaic numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if self.errorDatas(array, np.ndarray) or \
           self.errorDatas(dim, tuple, int) or \
           dim[0] <= 0 or dim[1] <= 0:
            return None
        tmpArray = self.juxtapose(array, dim[0], 0)
        tmpArray = self.juxtapose(tmpArray, dim[1], 1)
        return tmpArray


if __name__ == "__main__":
    # from ImageProcessor import ImageProcessor
    from srcs.ImageProcessor import ImageProcessor

    ip = ImageProcessor()
    image = ip.load("../ressources/42AI.png")
    scrap = ScrapBooker()
    # image2 = scrap.crop(image, (50, 50), (50, 50))

    # image2 = scrap.thin(image, 3, 1 )
    image2 = scrap.juxtapose(image, 3, 0)
    # image2 = scrap.mosaic(image, (3, 2))
    if image2 is not None:
        print("Loading image of dimensions {} x {}"
              .format(image2.shape[0], image2.shape[1]))
        ip.display(image2)
    spb = ScrapBooker()
    arr1 = np.arange(0, 25).reshape(5, 5)
    print(spb.crop(arr1, (3, 1), (1, 0)))
    arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1, 9)
    print(repr(spb.thin(arr2, 3, 0)))
    arr3 = np.array([[var] * 10 for var in "ABCDEFG"])
    print(repr(spb.thin(arr3, 3, 1)))
    arr4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(repr(spb.juxtapose(arr4, 2, 0)))
    not_numpy_arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    spb.crop(not_numpy_arr, (1, 2))
    spb.juxtapose(arr4, -2, 0)
    spb.mosaic(arr4, (1, 2, 3))
    arr3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(repr(spb.juxtapose(arr3, 3, 1)))
