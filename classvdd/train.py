import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from classvdd.model import autoencoder, network
from classvdd.utils.utils import weights_init_normal, EarlyStopping
from preprocess import get_ALeRCE_data


class TrainerClasSVDD:
    def __init__(self, args, dataloader_train, dataloader_val, device, scaler):
        self.args = args
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.device = device
        self.scaler = scaler
    

    def pretrain(self):
        """ Pretraining the weights for the ClasSVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _, _ in Bar(self.dataloader_train):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        torch.save({'net_dict': ae.state_dict()}, 'classvdd/weights/pretrained_parameters.pth')

    def train(self):
        """Training the ClasSVDD model"""
        if self.args.pretrain==True:
            self.load_pretrained_weights()
        else:
            self.net.apply(weights_init_normal)
            self.c = torch.randn(self.args.latent_dim).to(self.device)
        
        self.es = EarlyStopping(patience=self.args.patience)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        self.loss = []
        self.loss_t = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            self.net.train()
            for x, _, y in Bar(self.dataloader_train):
                x = x.float().to(self.device)
                y = y.long()

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c[y]) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training ClasSVDD... Epoch: {}, Loss: {:.3f}'.format(
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
            for x, _, y in Bar(self.dataloader_val):
                x = x.float().to(self.device)
                y = y.long()
                
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c[y]) ** 2, dim=1))
                total_loss+=loss.item()
                
        loss = total_loss/len(self.dataloader_val)
        print('Testing ClasSVDD... Epoch: {}, Loss: {:.3}'.format(
             epoch, loss
             ))
        stop = self.es.count(loss, self.net, self.c)
        return loss, stop
    
    def load_pretrained_weights(self):
        self.net = network().to(self.device)
        state_dict = torch.load('classvdd/weights/pretrained_parameters.pth')
        self.net.load_state_dict(state_dict['model'], strict=False)
        self.c = self.set_c().to(self.device)
       
    def load_weights(self):
        state_dict = torch.load('classvdd/weights/model_parameters.pth')
        self.net.load_state_dict(state_dict['model'])
        self.c = torch.Tensor(state_dict['center']).to(self.device)
    
    def set_c(self, eps=0.1):
        """Initializing the center for the hypersphere"""
        self.net.eval()
        dataloder, _, _ = get_ALeRCE_data(self.args.batch_size, 'train', mode='val')
        latents, labels = self.get_latent_space(dataloder)
        c = []
        for i in range(len(np.unique(labels))):
            ixs = np.where(labels == i)
            c.append(torch.mean(latents[ixs], dim=0))
        c = torch.stack(c)
        for i in range(len(c)):
            c[i][(abs(c[i]) < eps) & (c[i] < 0)] = -eps
            c[i][(abs(c[i]) < eps) & (c[i] > 0)] = eps
        return c

    def get_latent_space(self, dataloader):
        latents = []
        labels = []
        self.net.eval()
        with torch.no_grad():
            for x, _, y in dataloader:
                x, y = x.to(self.device).float(), y.long()
                z = self.net(x)
                latents.append(z.detach().cpu())
                labels.append(y)
        return torch.cat(latents), torch.cat(labels)
                

        

