import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

class ANN(nn.Module):

    def __init__(self, in_features, out_features):
        super(ANN, self).__init__()

        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, out_features)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)

        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, x, thresh=0.5):
        x = self(x)
        # return x.argmax(axis=-1)
        return (x > thresh).type(torch.int).reshape(-1)
    

    def train_model(self, X, y, epochs=1000, learning_rate=.001):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float)
            y = y.reshape(-1, 1)
            # oh = torch.zeros((*y.shape, 2))
            # oh[:, y] = 1
            # y = oh.type(torch.float)

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()  # sets the module in training mode
        for epoch in range(epochs):
            o = self(X)
            loss = criterion(o, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"{epoch} - loss: {loss}")

        self.eval()  # sets the module in evaluation mode

    def eval_model(self, X, y):
        self.eval()
        y_hat = self.predict(X)
        print(f"accuracy {accuracy_score(y, y_hat)}")
        print(f"precision {precision_score(y, y_hat)}")
        print(f"recall {recall_score(y, y_hat)}")
        print(f"confusion matrix \n {confusion_matrix(y, y_hat)}")








