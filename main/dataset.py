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
		self.images, self.labels = self.read_dataset(directory, dataset, mean)

		self.images = torch.from_numpy(self.images)
		self.labels = torch.from_numpy(self.labels)

	def __len__(self):
		return self.images.shape[0]

	def __getitem__(self, idx):
		sample = { "image": self.images[idx], "label": self.labels[idx] }

		return sample

	def read_dataset(self, directory, dataset="train", mean=None):
		if dataset == "train":
			filepath = os.path.join(directory, "train.p")
		elif dataset == "dev":
			filepath = os.path.join(directory, "dev.p")
		else:
			filepath = os.path.join(directory, "test.p")

		videos = pickle.load(open(filepath, "rb"))

		images = []
		labels = []
		for video in videos:
			for frame in video["frames"]:
				images.append(frame.reshape((1, 60, 80)))
				labels.append(CATEGORY_IDX[video["category"]])

		images = np.array(images, dtype=np.float32)
		labels = np.array(labels, dtype=np.uint8)

		if dataset == "train":
			self.mean = np.mean(images, axis=0)
			images -= self.mean
		elif mean is not None:
			images -= mean

		return images, labels

