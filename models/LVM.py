import torch
import torch.nn as nn
import torch.nn.functional as F

class LVMModel(nn.Module):
    def __init__(self, in_channels=90, seq_len=1000, latent_dim=64, num_classes=2):
        super(LVMModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate the size after convolutions and pooling
        conv_output_size = seq_len // (2**3)  # 3 maxpool layers with stride 2
        
        self.fc_mu = nn.Linear(512 * conv_output_size, latent_dim)
        self.fc_var = nn.Linear(512 * conv_output_size, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * conv_output_size),
            nn.ReLU(),
            nn.Unflatten(1, (512, conv_output_size)),
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        classification = self.classifier(z)
        return F.softmax(classification, dim=1), reconstructed, mu, logvar 