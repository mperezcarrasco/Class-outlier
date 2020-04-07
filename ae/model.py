import torch
import torch.nn as nn
import torch.nn.functional as F


class ae(nn.Module):
    def __init__(self, z_dim=32):
        super(ae, self).__init__()
        self.fc1 = nn.Linear(60, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, z_dim)
        self.fc5 = nn.Linear(z_dim, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, 60)

    def encode(self, x):
        h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = F.leaky_relu(self.bn2(self.fc2(h)))
        h = F.leaky_relu(self.bn3(self.fc3(h)))
        return self.fc4(h)

    def decode(self, x):
        h = F.leaky_relu(self.bn5(self.fc5(x)))
        h = F.leaky_relu(self.bn6(self.fc6(h)))
        h = F.leaky_relu(self.bn7(self.fc7(h)))
        return torch.sigmoid(self.fc8(h))
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
