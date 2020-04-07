import math
import torch
import numpy as np
from barbar import Bar

from torch import optim
import torch.nn.functional as F

from vade.forward_step import ComputeLosses
from vade.models import Autoencoder, VaDE
from vade.utils import EarlyStopping, get_priors
from preprocess import get_ALeRCE_data


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)

class TrainerVaDE:
    """This is the trainer for the Variational Deep Embedding (VaDE).
    """
    def __init__(self, args, dataloader_train, dataloader_test, device, n_classes, scaler):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.device = device
        self.args = args
        self.n_classes = n_classes
        self.scaler = scaler


    def pretrain(self):
        """Here we train an stacked autoencoder which will be used as the initialization for the VaDE. 
        This initialization is usefull because reconstruction in VAEs would be weak at the begining
        and the models are likely to get stuck in local minima.
        """
        ae = Autoencoder(latent_dim=self.args.latent_dim,
                         n_classes=self.n_classes).to(self.device)
        ae.apply(weights_init_normal)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _, _, _ in Bar(self.dataloader_train):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.dataloader_train)))
        torch.save({'net_dict': ae.state_dict()}, 'vade/weights/pretrained_parameters.pth')   


    def train(self):
        """Training the VaDE"""
        self.VaDE = VaDE(latent_dim=self.args.latent_dim,
                         n_classes=self.n_classes).to(self.device)
        if self.args.pretrain==True:
            self.load_pretrained_weights()
        else:
            self.VaDE.apply(weights_init_normal)

        self.es = EarlyStopping(patience=self.args.patience)

        self.optimizer = optim.Adam(self.VaDE.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=10, gamma=0.9)
        
        self.acc = []
        self.acc_t = []
        self.rec = []
        self.rec_t = []
        self.dkl = []
        self.dkl_t = []
        
        self.forward_step = ComputeLosses(self.VaDE)
        print('Training VaDE...')

        for epoch in range(self.args.num_epochs):
            self.train_VaDE(epoch)
            stop = self.test_VaDE(epoch)
            lr_scheduler.step()
            if stop:
                break
        self.load_weights()

    def train_VaDE(self, epoch):
        self.VaDE.train()
        total_loss = 0
        total_acc = 0
        total_dkl = 0
        total_rec = 0
        for x, _, y2 in Bar(self.dataloader_train):
            self.optimizer.zero_grad()
            x, y2 = x.to(self.device), y2.to(self.device).long()

            loss, reconst_loss, kl_div, acc = self.forward_step.forward('train', x, y2, epoch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_acc += acc
            total_dkl += kl_div.item()
            total_rec += reconst_loss.item()
        self.acc.append(total_acc/len(self.dataloader_train))
        self.dkl.append(total_dkl/len(self.dataloader_train))
        self.rec.append(total_rec/len(self.dataloader_train))
        print('Training VaDE... Epoch: {}, Loss: {:.3f}, Acc: {:.3f}'.format(epoch, 
            total_loss/len(self.dataloader_train), total_acc/len(self.dataloader_train)))


    def test_VaDE(self, epoch):
        self.VaDE.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            total_dkl = 0
            total_rec = 0
            for x, _, y2 in Bar(self.dataloader_test):
                x, y2 = x.to(self.device), y2.to(self.device).long()
                loss, reconst_loss, kl_div, acc = self.forward_step.forward('test', x, y2, epoch)
                total_loss += loss.item()
                total_acc += acc.item()
                total_dkl += kl_div.item()
                total_rec += reconst_loss.item()
        self.acc_t.append(total_acc/len(self.dataloader_test))
        self.dkl_t.append(total_dkl/len(self.dataloader_test))
        self.rec_t.append(total_rec/len(self.dataloader_test))
        print('Testing VaDE... Epoch: {}, Loss: {:.3f}, Acc: {:.3f}'.format(epoch, 
                total_loss/len(self.dataloader_test), total_acc/len(self.dataloader_test)))
        
        stop = self.es.count(total_rec/len(self.dataloader_test), self.VaDE)
        return stop

    def load_pretrained_weights(self):
        state_dict = torch.load('vade/weights/pretrained_parameters.pth')
        model = Autoencoder(latent_dim=self.args.latent_dim,
                            n_classes=self.n_classes).to(self.device)
        model.load_state_dict(state_dict['model'])
        dataloader, _, _ = get_ALeRCE_data(128, 'train', mode='val', scaler=self.scaler)
        pi, mean, var = get_priors(dataloader, model, self.device)
        self.VaDE.load_state_dict(state_dict['model'], strict=False)
        self.VaDE.pi_prior.data = pi.float().to(self.device)
        self.VaDE.mu_prior.data = mean.float().to(self.device)
        self.VaDE.log_var_prior.data = torch.log(var).float().to(self.device)

    def load_weights(self):
        state_dict = torch.load('vade/weights/model_parameters.pth')
        self.VaDE.load_state_dict(state_dict['model'])