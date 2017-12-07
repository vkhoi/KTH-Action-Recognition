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

        self.zero_center(mean)

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

    def zero_center(self, mean=None):
        if mean is None:
            mean = self.mean

        for i in range(len(self.instances)):
            self.instances[i] -= mean

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

        self.mean = np.mean(instances, axis=0)

        return instances, labels

class BlockFrameDataset(Dataset):
    def __init__(self, directory, dataset="train", mean=None):
        self.instances, self.labels = self.read_dataset(directory, 
            dataset, mean)

        self.zero_center(mean)

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

    def zero_center(self, mean=None):
        if mean is None:
            mean = self.mean

        for i in range(len(self.instances)):
            self.instances[i] -= mean

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

        self.mean = np.mean(instances, axis=0)

        return instances, labels

class BlockFrameFlowDataset(Dataset):
    def __init__(self, directory, dataset="train", mean=None):
        self.instances, self.labels = self.read_dataset(directory, 
            dataset, mean)

        self.zero_center(mean)

        for i in range(len(self.instances)):
            self.instances[i]["frames"] = torch.from_numpy(self.instances[i]["frames"])
            self.instances[i]["flow_x"] = torch.from_numpy(self.instances[i]["flow_x"])
            self.instances[i]["flow_y"] = torch.from_numpy(self.instances[i]["flow_y"])
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        sample = { 
            "instances": self.instances[idx], 
            "labels": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean=None):
        if mean is None:
            mean = self.mean

        for i in range(len(self.instances)):
            self.instances[i]["frames"] -= mean["frames"]
            self.instances[i]["flow_x"] -= mean["flow_x"]
            self.instances[i]["flow_y"] -= mean["flow_y"]


    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            frame_path = os.path.join(directory, "train.p")
            flow_path = os.path.join(directory, "train_flow.p")
        elif dataset == "dev":
            frame_path = os.path.join(directory, "dev.p")
            flow_path = os.path.join(directory, "dev_flow.p")
        else:
            frame_path = os.path.join(directory, "test.p")
            flow_path = os.path.join(directory, "test_flow.p")

        video_frames = pickle.load(open(frame_path, "rb"))
        video_flows = pickle.load(open(flow_path, "rb"))

        instances = []
        labels = []

        mean_frames = np.zeros((1, 15, 60, 80), dtype=np.float32)
        mean_flow_x = np.zeros((1, 14, 30, 40), dtype=np.float32)
        mean_flow_y = np.zeros((1, 14, 30, 40), dtype=np.float32)

        for i_video in range(len(video_frames)):
            current_block_frame = []
            current_block_flow_x = []
            current_block_flow_y = []

            frames = video_frames[i_video]["frames"]
            flow_x = [0] + video_flows[i_video]["flow_x"]
            flow_y = [0] + video_flows[i_video]["flow_x"]

            for i_frame in range(len(frames)):
                current_block_frame.append(frames[i_frame])

                if i_frame % 15 > 0:
                    current_block_flow_x.append(flow_x[i_frame])
                    current_block_flow_y.append(flow_y[i_frame])

                if (i_frame + 1) % 15 == 0:
                    current_block_frame = np.array(current_block_frame, dtype=np.float32).reshape((1, 15, 60, 80))
                    current_block_flow_x = np.array(current_block_flow_x, dtype=np.float32).reshape((1, 14, 30, 40))
                    current_block_flow_y = np.array(current_block_flow_y, dtype=np.float32).reshape((1, 14, 30, 40))

                    mean_frames += current_block_frame
                    mean_flow_x += current_block_flow_x
                    mean_flow_y += current_block_flow_y

                    instances.append({
                        "frames": current_block_frame,
                        "flow_x": current_block_flow_x,
                        "flow_y": current_block_flow_y
                    })

                    labels.append(CATEGORY_IDX[video_frames[i_video]["category"]])

                    current_block_frame = []
                    current_block_flow_x = []
                    current_block_flow_y = []

        mean_frames /= len(instances)
        mean_flow_x /= len(instances)
        mean_flow_y /= len(instances)

        self.mean = {
            "frames": mean_frames,
            "flow_x": mean_flow_x,
            "flow_y": mean_flow_y
        }

        labels = np.array(labels, dtype=np.uint8)

        # if dataset == "train":
        #     p = np.random.choice(len(instances), 256, replace=False)
        # else:
        #     p = np.random.choice(len(instances), 64, replace=False)

        # instances = [instances[i] for i in p]
        # labels = labels[p]

        return instances, labels