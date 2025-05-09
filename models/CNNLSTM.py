import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, in_channels=90, seq_len=1000, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        # (B, 64, L/2)
        x = x.permute(0, 2, 1)                      # (B, L/2, 64)
        out, _ = self.lstm(x)
        return F.softmax(self.fc(out[:, -1]), dim=1)