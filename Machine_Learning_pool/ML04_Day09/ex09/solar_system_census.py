import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLR
from utils import *

NB_LABELS = 4


try:
    dataX = pd.read_csv("solar_system_census.csv")
    dataY = pd.read_csv("solar_system_census_planets.csv")
    x = np.array(dataX[['height', 'weight', 'bone_density']])
    y = np.array(dataY[['Origin']])
    df_models = pd.read_csv("models.csv")
except (FileNotFoundError, KeyError):
    print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
    exit(1)

# find the best model
best = df_models.loc[df_models['F1Score'] == df_models["F1Score"].max()]
power = int(best.iloc[0]["power"])
alpha = float(best.iloc[0]["alpha"])
max_iter = int(best.iloc[0]["max_iter"])
lambda_ = float(best.iloc[0]['lambda'])

# train the best model
# add polynomial
ext_x = add_polynomial_features(x, power)

m, n = ext_x.shape
prediction = np.zeros((m, 1))

# split data
x_train, x_test, y_train, y_test = data_spliter(ext_x, y, 0.7)  # prop 0.7

# normalize dataset x
x_train, mu, std = zscore_normalization(x_train)
x_test = normalize_others(x_test, mu, std)

# train the model
all_theta = np.zeros((NB_LABELS, n + 1))

for K in range(NB_LABELS):
    thetas = np.ones((n + 1, 1))
    mylr = MyLR(thetas, alpha=alpha, lambda_=lambda_, max_iter=max_iter)
    y_zip_train = (y_train == K).astype(float)
    mylr = mylr.fit_(x_train, y_zip_train)
    all_theta[K, :] = mylr.theta.reshape((1, n + 1))


# avec x_test
X_test = mylr.add_intercept(x_test)
y_all_predict_test = mylr.sigmoid_(np.dot(X_test, np.transpose(all_theta)))
y_hat_test = np.argmax(y_all_predict_test, axis=1).reshape(x_test.shape[0], 1)
# print(bcolors.OK + "\nOn the test dataset {:.2f}% of accuracy"
#       .format(np.mean((y_hat_test == y_test).astype(float)) * 100)
#       + bcolors.RESET)

# calcul de F1Score avec le test set
f1score_test = 0
for i in range(NB_LABELS):
    f1score_test += f1_score_(y_test, y_hat_test, pos_label=i)
f1score_test /= NB_LABELS

#  prediction en probabilite de chaque label
x_norm = normalize_others(ext_x, mu, std)
X_norm = mylr.add_intercept(x_norm)
y_all_predict = mylr.sigmoid_(np.dot(X_norm, np.transpose(all_theta)))
# on selectionne l'index de la prediction la plus haute
prediction = np.argmax(y_all_predict, axis=1).reshape(m, 1)
# print(bcolors.OK + "On the whole dataset {:.2f}% of accuracy"
#       .format(np.mean((prediction == y).astype(float)) * 100)
#       + bcolors.RESET)
f1score = 0
for i in range(NB_LABELS):
    f1score += f1_score_(y, prediction, pos_label=i)
f1score /= NB_LABELS
# print(bcolors.OK + "F1 score of the whole set = {}".format(f1score))


###############################################################################
#               F1 SCORE OF MODELS AND OF TEST AND WHOLE SET                  #
###############################################################################
scores = np.array(df_models[['lambda', 'F1Score']])
m = scores.shape[0]
for i in range(m):
    print("for lambda = {}, F1Score = {}".format(scores[i][0], scores[i][1]))
print("With lambda = {}, F1Score of the test set = {}"
      .format(lambda_, f1score_test))
print("With lambda = {}, F1Score of the whole set = {}"
      .format(lambda_, f1score))

###############################################################################
#                             GRAPHIC PART                                    #
###############################################################################


###############################################################################
#                               BAR PLOT                                      #
###############################################################################

fig = df_models.plot.bar(x='lambda', y='F1Score', rot=0)
plt.show()


###############################################################################
#                          GRAPH ON WHOLE DATASET                             #
###############################################################################

venus = (prediction == 0).astype(int).nonzero()[0]
earth = (prediction == 1).astype(int).nonzero()[0]
mars = (prediction == 2).astype(int).nonzero()[0]
asteroid = (prediction == 3).astype(int).nonzero()[0]
true_venus = (y == 0).astype(int).nonzero()[0]
true_earth = (y == 1).astype(int).nonzero()[0]
true_mars = (y == 2).astype(int).nonzero()[0]
true_asteroid = (y == 3).astype(int).nonzero()[0]
plt.scatter(x[true_venus, 0], x[true_venus, 1], color="cyan",
            label="True Venus", marker="o")
plt.scatter(x[true_earth, 0], x[true_earth, 1], color="orange",
            label="True Earth", marker="o")
plt.scatter(x[true_mars, 0], x[true_mars, 1], color="red",
            label="True Mars", marker="o")
plt.scatter(x[true_asteroid, 0], x[true_asteroid, 1],
            color="blue", label="True Asteroid Belt", marker="o")
plt.scatter(x[venus, 0], x[venus, 1], color="cyan",
            label="Venus", marker=".")
plt.scatter(x[earth, 0], x[earth, 1], color="orange",
            label="Earth", marker=".")
plt.scatter(x[mars, 0], x[mars, 1], color="red",
            label="Mars", marker=".")
plt.scatter(x[asteroid, 0], x[asteroid, 1],
            color="blue", label="Asteroid Belt", marker=".")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()

plt.scatter(x[true_venus, 1], x[true_venus, 2], color="cyan",
            label="True Venus", marker="o")
plt.scatter(x[true_earth, 1], x[true_earth, 2], color="orange",
            label="True Earth", marker="o")
plt.scatter(x[true_mars, 1], x[true_mars, 2], color="red",
            label="True Mars", marker="o")
plt.scatter(x[true_asteroid, 1], x[true_asteroid, 2],
            color="blue", label="True Asteroid Belt", marker="o")
plt.scatter(x[venus, 1], x[venus, 2], color="cyan",
            label="Venus", marker=".")
plt.scatter(x[earth, 1], x[earth, 2], color="orange",
            label="Earth", marker=".")
plt.scatter(x[mars, 1], x[mars, 2], color="red",
            label="Mars", marker=".")
plt.scatter(x[asteroid, 1], x[asteroid, 2],
            color="blue", label="Asteroid Belt", marker=".")
plt.xlabel("Weight")
plt.ylabel("Bone Density")
plt.legend()
plt.show()

plt.scatter(x[true_venus, 0], x[true_venus, 2], color="cyan",
            label="True Venus", marker="o")
plt.scatter(x[true_earth, 0], x[true_earth, 2], color="orange",
            label="True Earth", marker="o")
plt.scatter(x[true_mars, 0], x[true_mars, 2], color="red",
            label="True Mars", marker="o")
plt.scatter(x[true_asteroid, 0], x[true_asteroid, 2],
            color="blue", label="True Asteroid Belt", marker="o")
plt.scatter(x[venus, 0], x[venus, 2], color="cyan",
            label="Venus", marker=".")
plt.scatter(x[earth, 0], x[earth, 2], color="orange",
            label="Earth", marker=".")
plt.scatter(x[mars, 0], x[mars, 2], color="red",
            label="Mars", marker=".")
plt.scatter(x[asteroid, 0], x[asteroid, 2],
            color="blue", label="Asteroid Belt", marker=".")
plt.xlabel("Height")
plt.ylabel("Bone Density")
plt.legend()
plt.show()
