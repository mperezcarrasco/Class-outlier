import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from vae.model import vae
from vae.utils import EarlyStopping


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)

class TrainerVAE:
    def __init__(self, args, train_loader, val_loader, device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.es = EarlyStopping(self.args.patience)

    def train(self):
        """Training VAE"""
        self.model = vae(self.args.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        self.reconst = []
        self.reconst_t = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            self.model.train()
            for x, _, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat, mu, log_var, _ = self.model(x)
                reconst_loss = F.mse_loss(x_hat, x, reduction='mean') * 1000
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * 0.001
                loss = reconst_loss + kl_div
                loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Training VAE... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
            self.reconst.append(total_loss/len(self.train_loader))
            loss_test, stop = self.test(epoch)
            self.reconst_t.append(loss_test)
            if stop:
                break
        self.load_weights()

    def test(self, epoch):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for x, _, _ in Bar(self.val_loader):
                x = x.float().to(self.device)
                x_hat, _, _, _ = self.model(x)
                reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
                total_loss+=reconst_loss.item()
        total_loss = total_loss/len(self.val_loader)
        print('Testing VAE... Epoch: {}, Loss: {:.3}'.format(
             epoch, total_loss
             ))
        stop = self.es.count(total_loss, self.model)
        return total_loss, stop

    def load_weights(self):
        state_dict = torch.load('vae/weights/model_parameters.pth')
        self.model.load_state_dict(state_dict['model'])
        
        






        
    


                

        

