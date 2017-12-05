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
		self.instances, self.labels = self.read_dataset(directory, dataset, mean)

		self.instances = torch.from_numpy(self.instances)
		self.labels = torch.from_numpy(self.labels)

	def __len__(self):
		return self.instances.shape[0]

	def __getitem__(self, idx):
		sample = { 
			"instances": self.instances[idx],
			"labels": self.labels[idx] 
		}

		return sample

	def read_dataset(self, directory, dataset="train", mean=None):
		if dataset == "train":
			filepath = os.path.join(directory, "train.p")
		elif dataset == "dev":
			filepath = os.path.join(directory, "dev.p")
		else:
			filepath = os.path.join(directory, "test.p")

		videos = pickle.load(open(filepath, "rb"))

		instances = []
		labels = []
		for video in videos:
			for frame in video["frames"]:
				instances.append(frame.reshape((1, 60, 80)))
				labels.append(CATEGORY_IDX[video["category"]])

		instances = np.array(instances, dtype=np.float32)
		labels = np.array(labels, dtype=np.uint8)

		if dataset == "train":
			self.mean = np.mean(instances, axis=0)
			instances -= self.mean
		elif mean is not None:
			instances -= mean

		if dataset == "train":
			p = np.random.choice(len(instances), 1024, replace=False)
		else:
			p = np.random.choice(len(instances), 256, replace=False)

		instances = instances[p]
		labels = labels[p]

		return instances, labels

class BlockFrameDataset(Dataset):
	def __init__(self, directory, dataset="train", mean=None):
		self.instances, self.labels = self.read_dataset(directory, 
			dataset, mean)

		self.instances = torch.from_numpy(self.instances)
		self.labels = torch.from_numpy(self.labels)

	def __len__(self):
		return self.instances.shape[0]

	def __getitem__(self, idx):
		sample = { 
			"instances": self.instances[idx], 
			"labels": self.labels[idx] 
		}

		return sample

	def read_dataset(self, directory, dataset="train", mean=None):
		if dataset == "train":
			filepath = os.path.join(directory, "train.p")
		elif dataset == "dev":
			filepath = os.path.join(directory, "dev.p")
		else:
			filepath = os.path.join(directory, "test.p")

		videos = pickle.load(open(filepath, "rb"))

		instances = []
		labels = []
		current_block = []
		for video in videos:
			for i, frame in enumerate(video["frames"]):
				current_block.append(frame)
				if len(current_block) % 15 == 0:
					current_block = np.array(current_block)
					instances.append(current_block.reshape((1, 15, 60, 80)))
					labels.append(CATEGORY_IDX[video["category"]])
					current_block = []

		instances = np.array(instances, dtype=np.float32)
		labels = np.array(labels, dtype=np.uint8)

		if dataset == "train":
			self.mean = np.mean(instances, axis=0)
			instances -= self.mean
		elif mean is not None:
			instances -= mean

		if dataset == "train":
			p = np.random.choice(len(instances), 1024, replace=False)
		else:
			p = np.random.choice(len(instances), 256, replace=False)

		instances = instances[p]
		labels = labels[p]

		return instances, labels