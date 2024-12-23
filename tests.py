from Supervised_learning.DL.CNN.CNN import CNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std for pre-trained models
])


data_dir = 'Supervised_learning/dataset/training_set'
val_dir = 'Supervised_learning/dataset/training_set'


dataset = ImageFolder(root=data_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)

batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)

cnn = CNN(next(iter(dataloader))[0][:1], out_features=2)

cnn.train_model(train_dataloader=dataloader, epochs=20)
cnn.eval_model(test_dataloader=val_dataloader)















# from Supervised_learning.DL.ANN.ANN import ANN
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# X = pd.read_csv("Supervised_learning/points.csv", delimiter=",", header=None).to_numpy()
# print(X.shape)

# X_train, X_test, Y_train, Y_test = train_test_split(X[:, :-1], X[:, -1], train_size=.8)

# ann = ANN(2, 1)

# ann.train_model(X_train, Y_train)
# ann.eval_model(X_test, Y_test)
# print(X_test.shape, Y_test.shape)

