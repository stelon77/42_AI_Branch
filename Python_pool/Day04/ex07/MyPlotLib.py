import pandas as pd
import matplotlib.pyplot as plt
from FileLoader import FileLoader
import scipy as sp
# from pylab import *


def checks(funct):
    def wrapper(*args, **kwargs):
        if (len(args) < 2 or not (isinstance(args[0],
           pd.DataFrame) and isinstance(args[1], list))):
            print("Wrong arguments used in function {}".format(funct.__name__))
            return None
        ret = funct(*args, **kwargs)
        return ret
    return wrapper


class MyPlotLib:
    def __init__(self):
        pass

    @staticmethod
    @checks
    def histogram(data, features, title=None):
        n_subplots, features = MyPlotLib._realFeaturesCheck(data, features)
        if n_subplots == 0:
            return
        if n_subplots == 1:
            fig, ax = plt.subplots(tight_layout=True)
            ax.set_title(features[0])
            ax.hist(data[features[0]])
            ax.grid()
        else:
            fig, axs = plt.subplots(1, n_subplots, tight_layout=True)
            for index, feature in enumerate(features):
                axs[index].set_title(feature)
                axs[index].hist(data[feature])
                axs[index].grid()
        plt.suptitle(title)
        plt.show()

    @staticmethod
    @checks
    def density(data, features, title=None):
        n_subplots, features = MyPlotLib._realFeaturesCheck(data, features)
        if n_subplots == 0:
            return
        fig, ax = plt.subplots(tight_layout=True)
        for feature in features:
            data[feature].plot(kind='density', label=feature)
        plt.legend(loc='upper right')
        plt.title(title)
        plt.show()

    @staticmethod
    @checks
    def box_plot(data, features, title=None):
        n_subplots, features = MyPlotLib._realFeaturesCheck(data, features)
        if n_subplots == 0:
            return
        data[features].boxplot(grid=False)
        plt.title(title)
        plt.show()

    @staticmethod
    @checks
    def pair_plot(data, features):
        n_subplots, features = MyPlotLib._realFeaturesCheck(data, features)
        if n_subplots == 0:
            return
        pd.plotting.scatter_matrix(data[features])
        plt.show()

    @staticmethod
    def _realFeaturesCheck(data, features):
        nb_subplot = 0
        real_features = []
        for feature in features:
            try:
                if ('int' in str(data[feature].dtypes)
                   or 'float' in str(data[feature].dtypes)):
                    nb_subplot += 1
                    real_features.append(feature)
            except KeyError:
                print(feature, "is not a dataFrame Key")
        if nb_subplot == 0:
            print("No numerical datas in the features arg")
        return nb_subplot, real_features


if __name__ == '__main__':
    loader = FileLoader()
    df = loader.load('../athlete_events.csv')
    # df = loader.load('../solar_system_census.csv')

    myplt = MyPlotLib()
    myplt.histogram(df, ['Height', 'medal'])
    myplt.histogram(df, ['Height', 'Medal'])
    myplt.histogram(df, ['Height'])
    myplt.histogram(df, ['Height', 'Weight'])
    myplt.histogram(df, ['Height', 'Weight', 'Age'])

    myplt.density(df, ['Height'])
    myplt.density(df, ['Height', 'Weight'])
    myplt.density(df, ['Height', 'Weight', 'Age'])

    myplt.pair_plot(df, ['Height'])
    myplt.pair_plot(df, ['Height', 'Weight'])
    myplt.pair_plot(df, ['Height', 'Weight', 'Age'])

    myplt.box_plot(df, ['Height'])
    myplt.box_plot(df, ['Height', 'Weight'])
    myplt.box_plot(df, ['Height', 'Weight', 'Age'])
