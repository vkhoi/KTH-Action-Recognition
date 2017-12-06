from dataset import *
from torch.autograd import Variable

import argparse
import torch
import torch.nn as nn

class CNNBlockFrame(nn.Module):
    def __init__(self):
        super(CNNBlockFrame, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(4, 5, 5)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(4, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.fc1 = nn.Linear(7680, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = nn.Dropout(0.5)(out)
        out = self.fc2(out)

        return out

