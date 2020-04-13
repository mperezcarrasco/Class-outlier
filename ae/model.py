import torch
import torch.nn as nn
import torch.nn.functional as F


class ae(nn.Module):
    def __init__(self, z_dim=32):
        super(ae, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 4, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(4)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, padding=3)
        self.bn4 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, padding=2)
        
    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x), negative_slope=0.1))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x), negative_slope=0.1))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
   
    def decode(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x, negative_slope=0.1), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x), negative_slope=0.1), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x), negative_slope=0.1), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
