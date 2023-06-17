import numpy as np
from sys import argv, maxsize
from random import sample
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Fonction pour la 3D
import math


def badArgs():  # OK
    print("arguments are not correct\n\nUsage:\nargs:\n\
-filepath=path_of_the_file\n\
-ncentroid=number_of_centroids\n\
-max_iter=maximum_number_of iterations\n")
    exit(1)


def parseArgvs(argv):
    length = len(argv)
    if length != 4:
        badArgs()
    args = []
    for i in range(1, length):
        if "=" not in argv[i]:
            badArgs()
        lst = argv[i].split("=", 1)
        args.append(lst)
    if (args[0][0] != "filepath" or args[1][0] != "ncentroid"
       or args[2][0] != "max_iter"):
        badArgs()
    path = args[0][1]
    try:
        n = int(args[1][1])
        maxIter = int(args[2][1])
    except ValueError:
        badArgs()
    if n <= 0 or maxIter <= 0:
        badArgs()
    return path, n, maxIter


def _display2D(header, datas, centroids, regions):
    colors = ['orange', 'blue', 'green', 'red',
              'black', 'fuchsia', 'silver', 'cyan', 'tomato', 'peru',
              'plum', 'navy', 'seagreen', 'pink', 'papayawhip']

    fig = plt.figure()
    ax1 = plt.subplot(221)
    for dot in datas:
        ax1.scatter(dot[0], dot[1], marker='.', color=colors[int(dot[3])])
    for index, centroid in enumerate(centroids):
        ax1.scatter(centroid[0], centroid[1], marker='o', color=colors[index])
    ax1.set_xlabel(header[0])
    ax1.set_ylabel(header[1])

    ax2 = plt.subplot(222)
    for dot in datas:
        ax2.scatter(dot[0], dot[2], marker='.', color=colors[int(dot[3])])
    for index, centroid in enumerate(centroids):
        ax2.scatter(centroid[0], centroid[2], marker='o', color=colors[index])
    ax2.set_xlabel(header[0])
    ax2.set_ylabel(header[2])


    ax3 = plt.subplot(223)
    for dot in datas:
        ax3.scatter(dot[1], dot[2], marker='.', color=colors[int(dot[3])])
    for index, centroid in enumerate(centroids):
        ax3.scatter(centroid[1], centroid[2], marker='o', color=colors[index])
    ax3.set_xlabel(header[1])
    ax3.set_ylabel(header[2])

    ax4 = plt.subplot(224, projection='3d')
    for point in datas:
        ax4.scatter(point[0], point[1], point[2], marker='.',
                    color=colors[int(point[3])])  # Trace des points 3D
    for index, centroid in enumerate(centroids):
        if len(regions) == 4:
            ax4.scatter(centroid[0], centroid[1], centroid[2], marker='o',
                        color=colors[index], label=regions[index])
        else:
            ax4.scatter(centroid[0], centroid[1], centroid[2], marker='o',
                        color=colors[index])
    plt.title("Solar system population")
    ax4.set_xlabel(header[0])
    ax4.set_ylabel(header[1])
    ax4.set_zlabel(header[2])
    if len(regions) == 4:
        ax4.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


class KmeansClustering:  # OK
    def __init__(self, max_iter=20, ncentroid=4):

        self.ncentroid = ncentroid  # number of centroids
        self.max_iter = max_iter  # number of maxiterations to update centroids
        self.centroids = []  # values of the centroids

    def fit(self, X):
        """
        Run the K-means clustering algorithm.
        For the location of the initial centroids,
        random pick ncentroids from the dataset.
        Args:
        X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Returns:
        None.
        Raises:
        This function should not raise any Exception.
        """
        if self.ncentroid > len(X):
            print("Too many centroids for this data array")
            exit
        normedX, mini, maxi = KmeansClustering.normalizeArray(X)

        choice = sample(range(0, X.shape[0]), self.ncentroid)
        for i in choice:
            self.centroids.append(normedX[i])
        # print(self.centroids)
        belongsTo = [-1 for i in range(len(normedX))]
        newMeans = self.centroids.copy()
        iteration = 0
        clusterSizes = []

        for nbIteration in range(self.max_iter):
            clusterSizes = [0 for i in range(len(self.centroids))]
            noChange = True

            for i in range(len(normedX)):
                dot = normedX[i]
                index = KmeansClustering.classify(self.centroids, dot)

                clusterSizes[index] += 1
                if(index != belongsTo[i]):
                    noChange = False
                belongsTo[i] = index

            # calcul nouvelle moyenne
            newMeans = (KmeansClustering.newMeanCalculation
                        (normedX, clusterSizes, belongsTo, newMeans))

            for nb, value in enumerate(newMeans):
                self.centroids[nb] = value
            if noChange:
                iteration = nbIteration + 1
                break
        self.centroids = (KmeansClustering.
                          deNormalizeCentroids(self.centroids, maxi, mini))

        # display the information
        if iteration:
            print("\ncentroids stable after {} iterations\n".format(iteration))
        else:
            print("\nafter {} iterations\n".format(self.max_iter))

    def predict(self, X):
        """
        Predict from wich cluster each datapoint belongs to.
        Args:
        X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Returns:
        the prediction has a numpy.ndarray, a vector of dimension m * 1.
        Raises:
        This function should not raise any Exception.
        """
        clusters = []
        for item in X:
            index = KmeansClustering.classify(self.centroids, item)
            clusters.append(index)
        cluster = np.array(clusters).reshape(len(clusters), 1)
        return cluster

    @staticmethod
    def normalizeArray(X):  # OK
        """normalize the array with values in interval [0, 1]"""
        normedX = np.copy(X)
        min = np.amin(X, axis=0)
        max = np.amax(X, axis=0)
        length = len(max)
        for line in normedX:
            for i in range(0, length):
                line[i] = (line[i] - min[i]) / (max[i] - min[i])
        return normedX, min, max

    @staticmethod
    def deNormalizeCentroids(X, max, min):
        for lst in X:
            for i in range(len(lst)):
                lst[i] = (lst[i] * (max[i] - min[i])) + min[i]
        return X

    @staticmethod
    def euclideanDistance(x, y):  # OK
        """return the euclidian distance between 2 points"""
        S = 0
        for i in range(len(x)):
            S += math.pow(x[i]-y[i], 2)
        return math.sqrt(S)

    @staticmethod
    def updateMean(n, mean, item):
        """update the mean gradually"""
        for i in range(len(mean)):
            m = mean[i]
            mean[i] = (m*(n-1)+item[i])/float(n)
        return mean

    @staticmethod
    def newMeanCalculation(normedX, clusterSizes, belongsTo, Means):
        newValues = [0 for i in range(len(clusterSizes))]
        for i, dot in enumerate(normedX):
            newValues[belongsTo[i]] += dot
        for i, size in enumerate(clusterSizes):
            if size == 0:
                newValues[i] = Means[i]  # ne change rien
            else:
                newValues[i] = newValues[i] / size  # nouvelle moyenne
        return newValues

    @staticmethod
    def classify(means, item):  # OK
        """ Classify item to the mean with minimum distance """
        minimum = maxsize
        index = -1
        for i in range(len(means)):
            dist = KmeansClustering.euclideanDistance(item, means[i])
            if (dist < minimum):
                minimum = dist
                index = i
        return index

    @staticmethod
    def pickTheRegion(centroids):
        """for ncentroid = 4, try to pick the region of each"""
        centers = np.array(centroids)
        indexArray = np.arange(4).reshape(4, 1)
        centers = np.concatenate((centers, indexArray), axis=1)
        regions = {}

        indexMax = np.argmax(centers, axis=0)
        indexOfMaxHeight = centers[indexMax[0]][3]
        regions[int(indexOfMaxHeight)] = "Asteroid Belt colonies"
        centers = np.delete(centers, indexMax[0], 0)

        indexMax = np.argmax(centers, axis=0)
        indexOfMaxHeight = centers[indexMax[0]][3]
        regions[int(indexOfMaxHeight)] = "Martian Republic"
        centers = np.delete(centers, indexMax[0], 0)

        indexMax = np.argmax(centers, axis=0)
        indexOfMaxWeight = centers[indexMax[1]][3]
        regions[int(indexOfMaxWeight)] = "United Nation of Earth"
        centers = np.delete(centers, indexMax[1], 0)

        regions[int(centers[0][3])] = "Flying cities of Venus"

        return(regions)


if __name__ == "__main__":
    from srcs.csvreader import *

    # parsing
    path, n, maxIter = parseArgvs(argv)
    # read the dataset and fit
    with CsvReader(path, header=True) as f:
        if f is None:
            print("bad file name or corrupted csv file")
            exit(1)
        header = f.getheader()[1:]
        datas = np.array(f.getdata()).astype(float)[:, 1:]
    km = KmeansClustering(maxIter, n)

    km.fit(datas)
    clusters = km.predict(datas)
    # a mettre dans une autre fonction
    # -----------------------------
    regions = {}
    print("The centroids are :")
    if n != 4:
        for index, centroid in enumerate(km.centroids):
            print("{} with {} individuals".format(centroid,
                  np.count_nonzero(clusters == index)))
    else:
        regions = km.pickTheRegion(km.centroids)
        for index, centroid in enumerate(km.centroids):
            print("{} with {} individuals : {}".format(centroid,
                  np.count_nonzero(clusters == index), regions[index]))
    # -----------------------------
    datas_to_draw = np.hstack((datas, clusters))
    if n < 16:
        _display2D(header, datas_to_draw, km.centroids, regions)
