from dataset import *
from torch.autograd import Variable

import argparse
import torch
import torch.nn as nn

class CNNBlockFrameFlow(nn.Module):
    def __init__(self):
        super(CNNBlockFrameFlow, self).__init__()

        self.conv1_frame = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(4, 5, 5)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_frame = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_frame = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.conv1_flow_x = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_flow_x = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_flow_x = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.conv1_flow_y = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_flow_y = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_flow_y = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.fc1 = nn.Linear(3328, 128)
        self.dropfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, frames, flow_x, flow_y):
        out_frames = self.conv1_frame(frames)
        out_frames = self.conv2_frame(out_frames)
        out_frames = self.conv3_frame(out_frames)
        out_frames = out_frames.view(out_frames.size(0), -1)

        out_flow_x = self.conv1_flow_x(flow_x)
        out_flow_x = self.conv2_flow_x(out_flow_x)
        out_flow_x = self.conv3_flow_x(out_flow_x)
        out_flow_x = out_flow_x.view(out_flow_x.size(0), -1)

        out_flow_y = self.conv1_flow_y(flow_y)
        out_flow_y = self.conv2_flow_y(out_flow_y)
        out_flow_y = self.conv3_flow_y(out_flow_y)
        out_flow_y = out_flow_y.view(out_flow_y.size(0), -1)

        out = torch.cat([out_frames, out_flow_x, out_flow_y], 1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropfc1(out)
        out = self.fc2(out)

        return out

