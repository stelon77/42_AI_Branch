import pandas as pd
import math


class bcolors:
    OK = '\033[92m'  # green
    WARNING = '\033[93m'  # yellow
    FAIL = '\033[91m'  # red
    RESET = '\033[0m'  # reset color


try:
    df = pd.read_csv('thetas.csv', sep=',')
    theta0 = df['theta0'][0]
    theta1 = df['theta1'][0]
    mu = df['mu'][0]
    std = df['std'][0]
except (FileNotFoundError, pd.errors.ParserError, KeyError):
    print(bcolors.FAIL
          + "csv datas corrupted or bad file name or bad data name"
          + bcolors.RESET)
    exit(1)

if std == 0:
    std = 1

if pd.isna(theta0) or pd.isna(theta1) \
   or math.isinf(theta0) or math.isinf(theta1):
    print(bcolors.FAIL
          + "theta is NaN, retry training with smaller alpha"
          + bcolors.RESET)
    exit(1)
run = True
while run:
    print("Please give a mileage to have a predicted price :")
    n = input(">> ")
    if n == 'exit':
        run = False
        continue
    try:
        nb = float(n)
    except ValueError:
        print(bcolors.FAIL
              + "You have to give a positive number or type 'exit'...try again"
              + bcolors.RESET)
        continue
    if nb < 0:
        print(bcolors.FAIL
              + "You have to give a positive number or type 'exit'...try again"
              + bcolors.RESET)
        continue

    #  mileage normalization  and  estimation
    nb = (nb - mu) / std
    predicted_price = theta0 + theta1 * nb

    if predicted_price < 0:
        print(bcolors.WARNING
              + 'for this mileage, the owner should give you money !!\n\n'
              + bcolors.RESET)
    else:
        print(bcolors.OK +
              "the estimated price is {:.2f}\n\n".format(predicted_price)
              + bcolors.RESET)
