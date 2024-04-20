from DL.ANN.ANN import ANN
from DL.CNN.CNN import CNN

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from torch.utils.data import DataLoader
import torchvision.datasets as vset
import torchvision.transforms as transforms

def run_ann():
    X = pd.read_csv("Supervised_learning/points.csv", delimiter=",", header=None).to_numpy()
    print(X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X[:, :-1], X[:, -1], train_size=.8)

    ann = ANN(2, 1)

    ann.train_model(X_train, Y_train)
    ann.eval_model(X_test, Y_test)
    print(X_test.shape, Y_test.shape)

def run_cnn():
    data_transform = transforms.Compose([ 
        transforms.ToTensor(),
    ])
    train_dataset = vset.CIFAR10(root='data/', 
        train=True,
        transform=data_transform,
        download=True
    )

    test_dataset = vset.CIFAR10(root='data/', 
        train=False,
        transform=data_transform,
        download=True
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    cnn = CNN()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        cnn.to(device)
        print(f"CNN in {device}")
    else:
        device = torch.device('cpu')


    cnn.train_model(train_dataloader=train_dataloader)
    cnn.eval_model(test_dataloader=test_dataloader)
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
        run_cnn()





