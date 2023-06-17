import pandas as pd


def howManyMedals(df, name):
    ret = {}
    name_df = df.loc[df['Name'] == name]
    years = name_df.groupby('Year')
    for year, datas in years:
        ret[year] = {'G': 0, 'S': 0, 'B': 0}
        for result in datas['Medal']:
            if result == 'Gold':
                ret[year]['G'] += 1
            if result == 'Silver':
                ret[year]['S'] += 1
            if result == 'Bronze':
                ret[year]['B'] += 1
    return ret


if __name__ == '__main__':
    from FileLoader import FileLoader

    loader = FileLoader()
    data = loader.load('../athlete_events.csv')
    print(repr(howManyMedals(data, "Kjetil Andr Aamodt")))
