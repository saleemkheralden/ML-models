import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from torch.utils.data import DataLoader
import torchvision.datasets as vset
import torchvision.transforms as transforms

# Before defining the CNN let's load the dataset for the train in the CNN class
# We'll load CIFAR10 dataset
# First thing we need the transforms, vset and DataLoader modules

# The main job of the transforms module is to transform the dataset from PIL.image.image to torch.Tensor
# We can in addition apply all sorts of transformations such as Rotations, normalization...

data_transform = transforms.Compose([ 
	# Examples of the transformations available

	# transforms.RandomHorizontalFlip(p=0.3),
	# transforms.RandomRotation(30),
	# transforms.ColorJitter(),
	# transforms.RandomVerticalFlip(p=0.3),
	# transforms.RandomCrop(32, padding=5),
	# transforms.Normalize((0.4914, 0.4822, 0.4465),
	# 						(0.247, 0.2434, 0.2615)),

	transforms.ToTensor(),
])

# we'll load the data
dataset = vset.CIFAR10(root='data/', 
	train=True,
	transform=data_transform,
	download=True
)

# now we put the data in dataloader that will handle the bathcing and getitem methods
dataloader = DataLoader(dataset=dataset, batch_size=20, shuffle=True)

# Now let's define a CNN
class CNN(nn.Module):
	def __init__(self, out_features=10):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))  # 3 x 32 x 32 -> 16 x 8 x 8

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),)  # 16 x 8 x 8 -> 32 x 8 x 8
			# nn.MaxPool2d(2))

		self.dropout1 = nn.Dropout(p=.3)

		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 40, kernel_size=3, padding=1),
			nn.BatchNorm2d(40),
			nn.ReLU(),
			nn.MaxPool2d(2))  # 32 x 8 x 8 -> 40 x 4 x 4

		self.layer4 = nn.Sequential(
			nn.Conv2d(40, 50, kernel_size=3, padding=1),
			nn.BatchNorm2d(50),
			nn.ReLU())  # 40 x 4 x 4 -> 50 x 4 x 4

		self.fc1 = nn.Linear(4 * 4 * 50, 15)  # 4 * 4 * 50 -> 15
		self.dropout = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(15, out_features)  # 15 -> 10
		self.logsoftmax = nn.LogSoftmax(dim=1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.dropout1(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = nn.functional.relu(out)
		out = self.dropout(out)
		out = self.fc2(out)
		return self.softmax(out)
		# return self.logsoftmax(out)

	def predict(self, x, thresh=0.5):
		self.eval()
		
		x = self(x)
		return x.argmax(axis=-1)
		# return (x > thresh).type(torch.int).reshape(-1)


	def train_model(self, X=None, y=None, train_dataloader=None, epochs=1000, learning_rate=.001):
		if (X is None) and (train_dataloader is None):
			raise Exception("Need either X or train_dataloader to not be None.")

		if (y is not None) and (not isinstance(y, torch.Tensor)):
			y = torch.tensor(y, dtype=torch.float)
			y = y.reshape(-1, 1)
			# oh = torch.zeros((*y.shape, 2))
			# oh[:, y] = 1
			# y = oh.type(torch.float)

		if train_dataloader is None:
			train_dataloader = zip(X, y)

		criterion = nn.CrossEntropyLoss()  # works best with softmax
		# criterion = nn.NLLLoss()

		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

		self.train()  # sets the module in training mode
		for epoch in range(epochs):

			for i, (x_batch, y_batch) in enumerate(train_dataloader):
				if torch.cuda.is_available():
					x_batch = x_batch.cuda()
					y_batch = y_batch.cuda()

				o = self(x_batch)
				loss = criterion(o, y_batch)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if torch.cuda.is_available():
					del x_batch
					del y_batch
					del o
					torch.cuda.empty_cache()

			if epoch % 100 == 0:
				print(f"{epoch} - loss: {loss}")

		self.eval()  # sets the module in evaluation mode

	def extract_y(self, dataloader):
		y = []
		y_hat = []

		with torch.no_grad():  # in evaluation there's no need to calculate the gradients, so we disable it.
			for input, target in dataloader:
				pred = self.predict(input)

				y_hat.extend(pred)
				y.extend(target)
		
		y = np.array(y)
		y_hat = np.array(y_hat)
		return y, y_hat

	def eval_model(self, X=None, y=None, test_dataloader=None):
		self.eval()
		if (X is None) and (y is None):
			y, y_hat = self.extract_y(test_dataloader)	
		else:
			y_hat = self.predict(X)

		print(f"accuracy {accuracy_score(y, y_hat)}")
		print(f"precision {precision_score(y, y_hat)}")
		print(f"recall {recall_score(y, y_hat)}")
		print(f"confusion matrix \n {confusion_matrix(y, y_hat)}")








