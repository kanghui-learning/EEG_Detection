import torch
import torch.nn as nn
import torch.nn.functional as F
# from numpy.lib.arraypad import pad


class FCNModel(nn.Module):
    def __init__(self, nchannel,nb_classes):
        super(FCNModel, self).__init__()
        # Conv1D expects (batch_size, channels, sequence_length)
        self.conv1 = nn.Conv1d(in_channels=nchannel, out_channels=128, kernel_size=8, padding=7)  # padding = (kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)  # padding = (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)  # padding = (kernel_size - 1) // 2
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool to output of size 1
        self.fc = nn.Linear(128, nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x should be of shape (batch_size, channels, sequence_length)
        # 64, 90, 1000
        x = torch.relu(self.bn1(self.conv1(x))) # [64, 128, 1007]
        x = torch.relu(self.bn2(self.conv2(x))) # [64, 256, 1007]
        x = torch.relu(self.bn3(self.conv3(x))) # [64, 128, 1007]
        x = self.global_avg_pool(x)             # [64, 128, 1]
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)                          # [64, 2]
        return self.softmax(x)