import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.linear_assignment_ import linear_assignment

class ComputeLosses:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def forward(self, mode, x_sup, y_sup, epoch):
        self.epoch=epoch
        
        reconst_loss, kl_div, clf_loss, probs = self.supervised_loss(x_sup, y_sup)
        loss = reconst_loss + kl_div + clf_loss
        acc = self.compute_metrics(y_sup, probs)

        return loss, reconst_loss, kl_div, acc*100

    def supervised_loss(self, x, y):
        x_hat, mu, log_var, z = self.model(x)

        means_batch = self.model.mu_prior[y, :]
        covs_batch = self.model.log_var_prior[y,:].exp()
        p_c = self.model.pi_prior

        kl_div = torch.mean((torch.sum(torch.log(2*np.pi*covs_batch) + \
                            torch.exp(log_var)/covs_batch + \
                            torch.pow(mu-means_batch,2)/covs_batch - \
                            (1+log_var),dim=1)*0.5 - z.size(1)*0.5*np.log(2*np.pi)))
        kl_div *= self.args.kl_mul

        reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
        reconst_loss *= self.args.rec_mul

        probs = self.compute_pcz(z, p_c)
        
        clf_loss = F.cross_entropy(probs, y, reduction='mean')
        clf_loss *= self.args.cl_mul
        
        reconst_loss *= 1
        kl_div *= 1
        clf_loss *= 1

        return reconst_loss, kl_div, clf_loss, probs
    
    def compute_pcz(self, z, p_c):
        covs = self.model.log_var_prior.exp()
        means = self.model.mu_prior

        h = (z.unsqueeze(1) - means).pow(2) / covs
        h += torch.log(2*np.pi*covs)
        p_z_c = torch.exp(torch.log(p_c + 1e-20).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-20
        p_z_given_c = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return p_z_given_c
    
    def compute_metrics(self, y, probs):
        y_pred = np.argmax(probs.cpu().detach().numpy(), axis=1)
        y_true = y.cpu().detach().numpy()
        
        acc = accuracy_score(y_pred, y_true)

        return acc
