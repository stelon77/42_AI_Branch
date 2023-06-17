import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


class ImageProcessor:

    def __init__(self):
        pass

    @staticmethod
    def load(path):
        """
        transform the image called with path in a numpy ndarray
        """
        try:
            image = img.imread(path)
            print("Loading image of dimensions {} x {}"
                  .format(image.shape[0], image.shape[1]))
            return image

        except Exception as e:
            if len(e.args) > 1:
                print("Exception : {} -- strerror: {}"
                      .format(e.__class__.__name__, e.args[1]))
            else:
                print("Exception : {} -- strerror: None"
                      .format(e.__class__.__name__))
            return None

    @staticmethod
    def display(array):
        """
        dysplays the .png image provided as a numpy array
        """
        if not isinstance(array, np.ndarray):
            print("the array provided is not a numpy.ndarray")
            return
        plt.axis('off')
        plt.imshow(array)
        plt.show()


if __name__ == '__main__':
    from ImageProcessor import ImageProcessor
    ip = ImageProcessor()
    image = ip.load("../42AI.png")
    # ip.load("loulou")
    # ip.load("../empty.png")
    ip.display(image)
