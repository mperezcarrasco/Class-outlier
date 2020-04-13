import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class VaDE(nn.Module):
    def __init__(self, latent_dim=10, n_classes=9, conv_dim=64):
        super(VaDE, self).__init__()
        
        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes, requires_grad=True)
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim), requires_grad=True)
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim), requires_grad=True)

        self.cnn1 = Conv(1, conv_dim, 5, 2, 2)
        self.cnn2 = Conv(conv_dim, conv_dim*2, 5, 2, 2)
        self.cnn3 = Conv(conv_dim*2, conv_dim*4, 5, 2, 2)
        self.cnn4 = Conv(conv_dim*4, conv_dim*8, 4, 1, 0)

        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(512, latent_dim)

        self.lin = nn.Linear(latent_dim, 512)
        self.cnn5 = Deconv(conv_dim*8, conv_dim*4, 4, 4, 0)
        self.cnn6 = Deconv(conv_dim*4, conv_dim*2, 4, 2, 1)
        self.cnn7 = Deconv(conv_dim*2, conv_dim, 4, 2, 2)
        self.cnn8 = Deconv(conv_dim, 1, 4, 2, 1, bn=False)

    def encode(self, x):
        h = F.leaky_relu(self.cnn1(x), 0.05)
        h = F.leaky_relu(self.cnn2(h), 0.05)
        h = F.leaky_relu(self.cnn3(h), 0.05)
        h = F.leaky_relu(self.cnn4(h), 0.05)
        h = h.view(-1, 512)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.lin(z)
        h = h.view(-1, 512, 1, 1)
        h = F.leaky_relu(self.cnn5(h), 0.05)
        h = F.leaky_relu(self.cnn6(h), 0.05)
        h = F.leaky_relu(self.cnn7(h), 0.05)
        return torch.sigmoid(self.cnn8(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

class Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True):
        super(Conv, self).__init__()
        self.bn = bn
        self.conv2d = nn.Conv2d(in_channels=dim_in, out_channels= dim_out,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=True)
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.conv2d(x))
        else:
            return self.conv2d(x)


class Deconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True):
        super(Deconv, self).__init__()
        self.bn = bn
        self.deconv2d = nn.ConvTranspose2d(in_channels=dim_in, out_channels=dim_out, 
                                           kernel_size=kernel_size, stride=stride, 
                                           padding=padding, bias=True) 
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.deconv2d(x))
        else: 
            return self.deconv2d(x)


"""
class VaDE(torch.nn.Module):
    def __init__(self, latent_dim=32, n_classes=10):
        super(VaDE, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes, requires_grad=True)
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim), requires_grad=True)
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim), requires_grad=True)
        
        self.z_dim = latent_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 4, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(4)
        
        self.mu = nn.Linear(4 * 7 * 7, latent_dim)
        self.log_var = nn.Linear(4 * 7 * 7, latent_dim)

        self.deconv1 = nn.ConvTranspose2d(1, 4, 5, padding=2)
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
        return self.mu(x), self.log_var(x)

    def decode(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x, negative_slope=0.1), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x), negative_slope=0.1), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x), negative_slope=0.1), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z
"""
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32, n_classes=10):
        super(Autoencoder, self).__init__()
        self.z_dim = latent_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 4, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * 7 * 7, latent_dim)

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