import pandas as pd


class SpatioTemporalData:

    def __init__(self, df=None):
        if not isinstance(df, pd.DataFrame):
            print("SpacioTemporalData needs a pandas.DataFrame \
as argument to be instanciated")
            exit(1)
        self.df = df

    def when(self, location):
        years = []
        loc_df = self.df.loc[self.df["City"] == location]
        town = loc_df.groupby('Year')
        for year, datas in town:
            years.append(year)
        return years

    def where(self, date):
        locations = []
        date_df = self.df.loc[self.df["Year"] == date]
        years = date_df.groupby('City')
        for location, datas in years:
            locations.append(location)
        return locations


if __name__ == '__main__':
    from FileLoader import FileLoader

    loader = FileLoader()
    data = loader.load('../athlete_events.csv')
    sp = SpatioTemporalData(data)
    print(sp.where(1896))
    print(sp.where(1895))
    print(sp.where(2016))
    print(sp.when('Athina'))
    print(sp.when('Paris'))
    print(sp.when('Lolo'))
