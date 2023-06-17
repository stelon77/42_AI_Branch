#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILE = "datasets/dataset_train.csv"

COURSES = ['Arithmancy', 'Astronomy', 'Herbology',
           'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
           'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
           'Care of Magical Creatures', 'Charms', 'Flying']


try:
    df = pd.read_csv(FILE)
except (FileNotFoundError):
    print("File or keys do not exist")
    exit(1)

df = df.drop(['Index', 'First Name', 'Last Name',
              'Birthday', 'Best Hand'], axis=1)
n = len(COURSES)

for i in range(0, n - 1):
    for j in range(i + 1, n):  # on tourne sur tous les features deux a deux
        x = np.array(df[[COURSES[i], COURSES[j]]])
        y = np.array(df['Hogwarts House'])
        griff = (y == 'Gryffindor').astype(int).nonzero()[0]
        huff = (y == 'Hufflepuff').astype(int).nonzero()[0]
        raven = (y == 'Ravenclaw').astype(int).nonzero()[0]
        slyt = (y == 'Slytherin').astype(int).nonzero()[0]
        plt.scatter(x[griff, 0], x[griff, 1], color="cyan",
                    label="Gryffindor", marker=".")
        plt.scatter(x[huff, 0], x[huff, 1], color="orange",
                    label="Hufflepuff", marker=".")
        plt.scatter(x[raven, 0], x[raven, 1], color="red",
                    label="Ravenclaw", marker=".")
        plt.scatter(x[slyt, 0], x[slyt, 1],
                    color="blue", label="Slytherin", marker=".")
        plt.xlabel(COURSES[i])
        plt.ylabel(COURSES[j])
        plt.legend()
        plt.show()
