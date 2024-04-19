from DL.ANN.ANN import ANN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def run_ann():
    X = pd.read_csv("Supervised_learning/points.csv", delimiter=",", header=None).to_numpy()
    print(X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X[:, :-1], X[:, -1], train_size=.8)

    ann = ANN(2, 1)

    ann.train_model(X_train, Y_train)
    ann.eval_model(X_test, Y_test)
    print(X_test.shape, Y_test.shape)


if __name__ == "__main__":
    argv = sys.argv
    if len(sys.argv) != 2:
        print(f"Must run python file with the neural network as a param, Ex. (python runner.py ANN)")
        print(f"[ANN, CNN, RNN, LSTM, ]")
        exit(1)

    model = argv[1]
    if model == "ANN":
        run_ann()
    elif model == "CNN":
        print("CNCENCNE")





