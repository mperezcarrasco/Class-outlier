import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from ae.model import ae
from ae.utils import EarlyStopping


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)

class TrainerAE:
    def __init__(self, args, train_loader, val_loader, device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.es = EarlyStopping(self.args.patience, self.args)

    def train(self):
        """Training the autoencoder"""
        self.model = ae(self.args.latent_dim).to(self.device)
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
                x_hat, _ = self.model(x)
                reconst_loss = F.mse_loss(x_hat, x, reduction='mean')

                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Training Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
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
                x_hat, _ = self.model(x)
                reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
                total_loss+=reconst_loss.item()
        loss = total_loss/len(self.val_loader)
        print('Testing Autoencoder... Epoch: {}, Loss: {:.3}'.format(
             epoch, loss
             ))
        stop = self.es.count(loss, self.model)
        return loss, stop
   
    def load_weights(self):
        state_dict = torch.load('ae/weights/model_parameters_{}.pth'.format(
                                self.args.anormal_class))
        self.model.load_state_dict(state_dict['model'])
        






        
    


                

        

