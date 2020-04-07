import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience):
        """Class for determining the stopping criterion for the model"""
        self.patience = patience
        self.counter = 0
        self.best_loss = 9999

    def count(self, loss, model):
        is_best = bool( loss <= self.best_loss)
        self.best_loss = min(loss, self.best_loss)
        if is_best:
            self.counter = 0
            self.save_weights(model)
            print('Weights saved.')
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
        else:
            return False
 
    def save_weights(self, model):
        """Save VaDE weights."""
        torch.save({'model': model.state_dict()}, 'vade/weights/model_parameters.pth')
    
def get_priors(dataloader, model, device):
    latent, labels = get_latent_space(dataloader, model, device)
    mean = []
    var = []
    proportion = []
    for i in range(len(np.unique(labels))):
        ixs = np.where(labels == i)
        mean.append(torch.mean(latent[ixs], dim=0))
        var.append(torch.std(latent[ixs], dim=0)**2)
        proportion.append(len(ixs))
    mean = torch.stack(mean)
    var = torch.stack(var)
    proportion = torch.Tensor(proportion)/torch.sum(torch.Tensor(proportion))
    return proportion, mean, var 

def get_latent_space(dataloader, model, device):
    latents = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, _, y in dataloader:
            x, y = x.to(device).float(), y.long()
            _, z = model(x)
            latents.append(z.detach().cpu())
            labels.append(y)
    return torch.cat(latents), torch.cat(labels)