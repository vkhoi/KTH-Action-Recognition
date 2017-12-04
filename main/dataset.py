import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset, DataLoader

CATEGORY_IDX = {
	"boxing": 0,
	"handclapping": 1,
	"handwaving": 2,
	"jogging": 3,
	"running": 4,
	"walking": 5
}

class RawDataset(Dataset):
	def __init__(self, directory, dataset="train", mean=None):
		self.images, self.labels = self.read_dataset(directory, dataset)

		if dataset == "train":
			self.mean = np.mean(self.images, axis=0)
		elif mean is not None:
			self.images -= mean

		self.images = torch.from_numpy(self.images)
		self.labels = torch.from_numpy(self.labels)

	def __len__(self):
		return self.images.shape[0]

	def __getitem__(self, idx):
		sample = { "image": self.images[idx], "label": self.labels[idx] }

		return sample

	def read_dataset(self, directory, dataset="train"):
		if dataset == "train":
			X = os.path.join(directory, "X_train.p")
			y = os.path.join(directory, "y_train.p")
		elif dataset == "dev":
			X = os.path.join(directory, "X_dev.p")
			y = os.path.join(directory, "y_dev.p")
		else:
			X = os.path.join(directory, "X_test.p")
			y = os.path.join(directory, "y_test.p")

		# Need dummy last channel for conv2d layer.
		images = pickle.load(open(X, "rb"))
		for i in range(len(images)):
			images[i] = images[i].reshape((1, 60, 80))
		images = np.array(images, dtype=np.float32)

		labels = pickle.load(open(y, "rb"))
		for i in range(len(labels)):
			labels[i] = CATEGORY_IDX[labels[i]]
		labels = np.array(labels, dtype=np.uint8)

		return images, labels

