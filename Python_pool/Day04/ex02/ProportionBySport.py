import pandas as pd


def proportionBySport(df, year, sport, sex):

    df_gender = df.loc[(df['Year'] == year) & (df['Sex'] == sex)]
    # exemple du sujet exact
    nb_by_sport = (df_gender.loc[(df_gender['Sport']) == sport]
                   .drop_duplicates(subset='Name').shape[0])
    nb_gender = df_gender.drop_duplicates(subset='Name').shape[0]

    # # pour etre tout a fait exact, on enleve doublons par ID
    # nb_by_sport = (df_gender.loc[(df_gender['Sport']) == sport]
    #                .drop_duplicates(subset='ID').shape[0])
    # nb_gender = df_gender.drop_duplicates(subset='ID').shape[0]

    # # exemple de la correction, ne checke pas doublons
    # nb_by_sport = df_gender.loc[(df_gender['Sport']) == sport].shape[0]
    # nb_gender = df_gender.shape[0]

    try:
        return nb_by_sport / nb_gender
    except ZeroDivisionError:
        print("this is not an olympic year")
        return None


if __name__ == '__main__':
    from FileLoader import FileLoader

    loader = FileLoader()
    data = loader.load('../athlete_events.csv')
    print(proportionBySport(data, 2004, 'Tennis', 'F'), end="\n\n")
    # print(proportionBySport(data, 2008, 'Hockey', 'F'), end="\n\n")
    # print(proportionBySport(data, 1964, 'Biathlon', 'M'), end="\n\n")
    # print(proportionBySport(data, 1963, 'Biathlon', 'M'), end="\n\n")
