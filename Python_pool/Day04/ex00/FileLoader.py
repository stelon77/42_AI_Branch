import pandas as pd


class FileLoader:

    def __init__(self):
        pass

    @staticmethod
    def load(path):
        """
        function which takes the file path of the dataset to load,
        displays a message specifying the dimensions of the dataset
        and returns the dataset loaded as a pandas.DataFrame

        argument :
        the file path of the dataset to load, displays a
        return :
        a pandas.DataFrame
        """
        if type(path) is not str:
            print("path has to be a string")
            return None
        try:
            new = pd.read_csv(path)
        except IOError:
            print("wrong path or wrong file")
            return None
        print(("loading dataset of dimensions {} x {}".
               format(new.shape[0], new.shape[1])))
        return new

    @staticmethod
    def display(df, n):
        """
        displays the first n rows of the dataset if n is positive,
        or the last n rows if n is negative.

        arguments:
        a pandas.DataFrame and an integer
        """
        if not isinstance(df, pd.DataFrame) or not isinstance(n, int):
            return None
        if n >= 0:
            print(df.head(n))
        else:
            print(df.tail(-n))


if __name__ == "__main__":
    fl = FileLoader()
    dt = fl.load("../athlete_events.csv")
    # fl.display(dt,7)
    # fl.display(dt, 0)
    fl.display(dt, -4)
