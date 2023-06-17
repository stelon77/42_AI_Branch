#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE = "datasets/dataset_train.csv"

COURSES = ['Arithmancy', 'Astronomy', 'Herbology',
           'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
           'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
           'Care of Magical Creatures', 'Charms', 'Flying']
# HOUSES = ['Griffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

# recuperer donnees
try:
    df = pd.read_csv(FILE)
except (FileNotFoundError):
    print("File or keys do not exist")
    exit(1)

# on supprime colonnes non utiles
df = df.drop(['Index', 'First Name', 'Last Name',
              'Birthday', 'Best Hand'], axis=1)
for course in COURSES:
    work_df = df[['Hogwarts House', course]]
    home_df = work_df.groupby('Hogwarts House')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    colors = ['r', 'g', 'b', 'y']
    data_list = []
    house_list = []
    for home, data in home_df:
        data_list.append(list(data[course]))
        house_list.append(home)

    ax.hist(data_list, density=True, histtype='bar', color=colors,
            label=house_list)
    ax.legend()
    ax.set_title(course)
    plt.show()
