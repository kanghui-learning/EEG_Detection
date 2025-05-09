import torch.nn as nn
import torch.nn.functional as F

class BiGRUModel(nn.Module):
    def __init__(self, in_channels=90, seq_len=1000, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # For bidirectional GRU, the output size is hidden_dim * 2
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        
        # Forward propagate GRU
        out, _ = self.gru(x)
        
        # Take the last time step's output
        out = out[:, -1, :]
        
        # Apply fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return F.softmax(out, dim=1) 