import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self, in_channels=90, seq_len=1000, num_classes=2):
        super(DeepConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 25, kernel_size=5),
            nn.BatchNorm1d(25),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(25, 50, kernel_size=5),
            nn.BatchNorm1d(50),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(50, 100, kernel_size=5),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(100, 200, kernel_size=5),
            nn.BatchNorm1d(200),
            nn.ELU(),
            nn.MaxPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return F.softmax(self.fc(x), dim=1)