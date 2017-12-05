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

		p = np.random.choice(len(images), 1024, replace=False)
		images = images[p]
		labels = labels[p]

		return images, labels

class BlockFrameDataset(Dataset):
	def __init__(self, directory, dataset="train", mean=None):
		self.block_frames, self.labels = self.read_dataset(directory, 
			dataset, mean)

		self.block_frames = torch.from_numpy(self.block_frames)
		self.labels = torch.from_numpy(self.labels)

	def __len__(self):
		return self.block_frames.shape[0]

	def __getitem__(self, idx):
		sample = { 
			"block_frames": self.block_frames[idx], 
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

		block_frames = []
		labels = []
		current_block = []
		for video in videos:
			for i, frame in enumerate(video["frames"]):
				current_block.append(frame)
				if len(current_block) % 15 == 0:
					current_block = np.array(current_block)
					block_frames.append(current_block.reshape((1, 15, 60, 80)))
					labels.append(CATEGORY_IDX[video["category"]])
					current_block = []

		block_frames = np.array(block_frames, dtype=np.float32)
		labels = np.array(labels, dtype=np.uint8)

		if dataset == "train":
			self.mean = np.mean(block_frames, axis=0)
			block_frames -= self.mean
		elif mean is not None:
			block_frames -= mean

		p = np.random.choice(len(block_frames), 512, replace=False)
		block_frames = block_frames[p]
		labels = labels[p]

		return block_frames, labels