import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from MyPlotLib import MyPlotLib
from FileLoader import FileLoader


class Komparator:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            return
        self.df = df

    def _generate_datas(self, categorical_var, numerical_var):
        mydatas = pd.DataFrame()
        df = self.df.loc[:, [categorical_var, numerical_var]]
        categories = df.groupby(categorical_var)
        for category, datas in categories:
            new_df = (datas.reset_index()
                      .rename(columns={numerical_var: category})
                      .loc[:, [category]])
            mydatas = pd.concat([mydatas, new_df])
        list_of_index = mydatas.columns.tolist()
        return mydatas, list_of_index

    def _check_vars(self, cat_data, num_data):
        if (cat_data not in self.df.select_dtypes(include='object') or
           num_data not in self.df.select_dtypes(exclude='object')):
            print("wrong type of variable")
            return False
        return True

    def compare_box_plots(self, categorical_var, numerical_var):
        if not Komparator._check_vars(self, categorical_var, numerical_var):
            return
        data, features = (Komparator._generate_datas
                          (self, categorical_var, numerical_var))
        mpl = MyPlotLib()
        mpl.box_plot(data, features, numerical_var)

    def density(self, categorical_var, numerical_var):
        if not Komparator._check_vars(self, categorical_var, numerical_var):
            return
        data, features = (Komparator._generate_datas
                          (self, categorical_var, numerical_var))
        mpl = MyPlotLib()
        mpl.density(data, features, numerical_var)

    def compare_histograms(self, categorical_var, numerical_var):
        if not Komparator._check_vars(self, categorical_var, numerical_var):
            return
        data, features = (Komparator._generate_datas
                          (self, categorical_var, numerical_var))
        mpl = MyPlotLib()
        mpl.histogram(data, features, numerical_var)


if __name__ == '__main__':
    loader = FileLoader()
    df = loader.load('../athlete_events.csv')
    # df = loader.load('../solar_system_census.csv')
    k = Komparator(df)
    k.compare_box_plots('Medal', 'Age')
    k.compare_box_plots('Sex', 'Height')

    k.density('Medal', 'Age')
    k.density('Sex', 'Height')
    # should not give results
    k.compare_histograms('Medal', 'Sex')
    k.compare_histograms('Height', 'Age')

    k.compare_histograms('Medal', 'Age')
    k.compare_histograms('Sex', 'Height')
