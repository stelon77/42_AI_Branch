import pandas as pd


def youngestfellah(df, year):
    """
    Get the name of the youngest woman and man for the given year.

    Args:
    df: pandas.DataFrame object containing the dataset.
    year: integer corresponding to a year.
    Returns:
    dct: dictionary with 2 keys for female and male athlete.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(year, int):
        print("wrong arguments for youngfellah function")
        return None
    minF = (df.loc[(df['Year'] == year) & (df['Sex'] == 'F')])["Age"].min()
    minM = (df.loc[(df['Year'] == year) & (df['Sex'] == 'M')])["Age"].min()
    return {'F': minF, 'M': minM}


if __name__ == "__main__":
    from FileLoader import FileLoader

    fl = FileLoader()
    df = fl.load("../athlete_events.csv")
    print(youngestfellah(df, 1992))
