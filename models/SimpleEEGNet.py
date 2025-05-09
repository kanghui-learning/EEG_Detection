import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimpleEEGNet(nn.Module):
    def __init__(self, in_channels=90, num_classes=2):
        super(SimpleEEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(2),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, groups=32, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.global_pool(x).squeeze(-1)
        return F.softmax(self.fc(x), dim=1)
