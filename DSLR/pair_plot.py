#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FILE = "datasets/dataset_train.csv"

COLUMNS = ['Hogwarts House', 'Ari', 'Astro', 'Herb',
           'DADA', 'Div', 'MS',
           'AR', 'Hist', 'Trans', 'Po',
           'CMC', 'Ch', 'Fly']

try:
    df = pd.read_csv(FILE)
except (FileNotFoundError):
    print("File or keys do not exist")
    exit(1)

df = df.drop(['Index', 'First Name', 'Last Name',
              'Birthday', 'Best Hand'], axis=1)
df.columns = COLUMNS
plt.figure()
sns.pairplot(df, hue='Hogwarts House')
plt.show()
