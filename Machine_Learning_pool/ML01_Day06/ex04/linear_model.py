import pandas as pd
import numpy as np
# from sklearn.metrics import mean_squared_error
from tools.my_linear_regression import MyLinearRegression as MyLR

FILENAME = "are_blue_pills_magics.csv"
X_KEY = "Micrograms"
Y_KEY = "Score"

# read dataset from csv file
# and extract datas
try:
    data = pd.read_csv(FILENAME)
    Xpill = np.array(data[X_KEY]).reshape(-1, 1)
    Yscore = np.array(data[Y_KEY]).reshape(-1, 1)
except (FileNotFoundError, KeyError):
    print("File or keys do not exist")
    exit(1)

# perform a linear regression
# and print results (thetas and MSE)
# at each example : display plots
linear_model1 = MyLR(np.array([[1.], [1.]]), alpha=5e-4, max_iter=100000)
Y_model1 = linear_model1.predict_(Xpill)
print("Example1:\nOriginals thetas: theta[0] = 1, theta[1] = 1")
print("Original MSE =", linear_model1.mse_(Yscore, Y_model1))
linear_model1.data_hypothesis_graph(Xpill, Yscore, Y_model1)
print("\n########################################################\n")
linear_model1.fit_(Xpill, Yscore)
print("Trained thetas: theta[0] = {:f}, theta[1] = {:f}"
      .format(linear_model1.thetas[0][0], linear_model1.thetas[1][0]))
Y_model1_1 = linear_model1.predict_(Xpill)
print("Trained MSE = {}\n".format(linear_model1.mse_(Yscore, Y_model1_1)))
linear_model1.data_hypothesis_graph(Xpill, Yscore, Y_model1_1)
linear_model1.cost_function_graph(Xpill, Yscore)


linear_model2 = MyLR(np.array([[89.], [-6.]]))
Y_model2 = linear_model2.predict_(Xpill)
print("Example2:\nOriginals thetas: theta[0] = {}, theta[1] = {}"
      .format(linear_model2.thetas[0][0], linear_model2.thetas[1][0]))
print("Original MSE =", linear_model2.mse_(Yscore, Y_model2))
# print(mean_squared_error(Yscore, Y_model2))
linear_model2.data_hypothesis_graph(Xpill, Yscore, Y_model2)

print("\n########################################################\n")
linear_model2.fit_(Xpill, Yscore)
print("Trained thetas: theta[0] = {:f}, theta[1] = {:f}"
      .format(linear_model2.thetas[0][0], linear_model2.thetas[1][0]))
Y_model2_1 = linear_model2.predict_(Xpill)
print("Trained MSE = {}\n".format(linear_model2.mse_(Yscore, Y_model2_1)))
linear_model2.data_hypothesis_graph(Xpill, Yscore, Y_model2_1)
linear_model2.cost_function_graph(Xpill, Yscore)
