import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from deepsvdd.model import autoencoder, network
from deepsvdd.utils.utils import weights_init_normal, EarlyStopping


class TrainerDeepSVDD:
    def __init__(self, args, dataloader_train, dataloader_val, device):
        self.args = args
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.device = device
    

    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        self.ae = autoencoder(self.args.latent_dim).to(self.device)
        self.ae.apply(weights_init_normal)
        optimizer = optim.Adam(self.ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=self.args.lr_milestones_ae, gamma=0.1)
        
        self.ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _, _ in Bar(self.dataloader_train):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = self.ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.dataloader_train)))
        torch.save({'model': self.ae.state_dict()}, 'deepsvdd/weights/pretrained_parameters_{}.pth'.format(
                                                     self.args.anormal_class))

    def train(self):
        """Training the Deep SVDD model"""
        if self.args.pretrain==True:
            self.load_pretrained_weights()
        else:
            self.net.apply(weights_init_normal)
            self.c = torch.randn(self.args.latent_dim).to(self.device)
        
        self.es = EarlyStopping(patience=self.args.patience)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones_ae, gamma=0.1)
        self.loss = []
        self.loss_t = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            self.net.train()
            for x, _, _ in Bar(self.dataloader_train):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.dataloader_train)))
            
            self.loss.append(total_loss/len(self.dataloader_train))
            loss_test, stop = self.test(epoch)
            self.loss_t.append(loss_test)
            if stop:
                break
        self.load_weights()

    def test(self, epoch):
        self.net.eval()

        total_loss = 0
        with torch.no_grad():
            for x, _, _ in Bar(self.dataloader_val):
                x = x.float().to(self.device)
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))

                total_loss+=loss.item()
        loss = total_loss/len(self.dataloader_val)
        print('Testing Deep SVDD... Epoch: {}, Loss: {:.3}'.format(
             epoch, loss
             ))
        stop = self.es.count(loss, self.net, self.c, self.args)
        return loss, stop
    
    def load_pretrained_weights(self):
        self.net = network().to(self.device)
        state_dict = torch.load('deepsvdd/weights/pretrained_parameters_{}.pth'.format(self.args.anormal_class))
        self.net.load_state_dict(state_dict['model'], strict=False)
        self.c = self.set_c().to(self.device)
       
    def load_weights(self):
        state_dict = torch.load('deepsvdd/weights/model_parameters_{}.pth'.format(self.args.anormal_class))
        self.net.load_state_dict(state_dict['model'])
        self.c = torch.Tensor(state_dict['center']).to(self.device)
    
    def set_c(self, eps=0.1):
        """Initializing the center for the hypersphere"""
        self.net.eval()
        z_ = []
        with torch.no_grad():
            for x, _, _ in self.dataloader_train:
                x = x.float().to(self.device)
                z = self.net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c
                

        

