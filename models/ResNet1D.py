import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet1D(nn.Module):
    def __init__(self, in_channels=90, num_classes=2):
        super(ResNet1D, self).__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,1), stride=(2,1), padding=(3,0), bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (B, 90, 1000)
        x = x.unsqueeze(1)  # (B, 1, 90, 1000)
        return F.softmax(self.resnet(x), dim=1)