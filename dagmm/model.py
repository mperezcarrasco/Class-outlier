import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMM, self).__init__()
        #Encoder network
        self.fc1 = nn.Linear(60, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, z_dim)
        self.fc5 = nn.Linear(z_dim, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, 60)

        #Estimation network
        self.fc9 = nn.Linear(z_dim+2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

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
    
    def estimate(self, z):
        h = F.dropout(F.leaky_relu(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma