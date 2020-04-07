import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class VaDE(torch.nn.Module):
    def __init__(self, latent_dim=10, n_classes=10):
        super(VaDE, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes, requires_grad=True)
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim), requires_grad=True)
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim), requires_grad=True)
        
        self.fc1 = nn.Linear(60, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.mu = nn.Linear(64, latent_dim) #Latent mu
        self.log_var = nn.Linear(64, latent_dim) #Latent logvar

        self.fc5 = nn.Linear(latent_dim, 64)
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
        return self.mu(h), self.log_var(h)

    def decode(self, z):
        h = F.leaky_relu(self.bn5(self.fc5(z)))
        h = F.leaky_relu(self.bn6(self.fc6(h)))
        h = F.leaky_relu(self.bn7(self.fc7(h)))
        return F.sigmoid(self.fc8(h))

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z


class Autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
        
    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
   
    def decode(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z