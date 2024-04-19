from Supervised_learning.DL.ANN.ANN import ANN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv("Supervised_learning/points.csv", delimiter=",", header=None).to_numpy()
print(X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X[:, :-1], X[:, -1], train_size=.8)

ann = ANN(2, 1)

ann.train_model(X_train, Y_train)
ann.eval_model(X_test, Y_test)
print(X_test.shape, Y_test.shape)

