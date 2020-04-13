import torch


class EarlyStopping:
    def __init__(self, patience, args):
        """Class for determining the stopping criterion for the model"""
        self.patience = patience
        self.counter = 0
        self.best_loss = 9999
        self.args = args

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
        """Save VAE weights."""
        torch.save({'model': model.state_dict()}, 'vae/weights/model_parameters_{}.pth'.format(
                                                   self.args.anormal_class))