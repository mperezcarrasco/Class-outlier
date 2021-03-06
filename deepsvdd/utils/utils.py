import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data, 0.0, 0.02)

class EarlyStopping:
    def __init__(self, patience):
        """Class for determining the stopping criterion for the model"""
        self.patience = patience
        self.counter = 0
        self.best_loss = 9999

    def count(self, loss, model, c, args):
        is_best = bool(loss <= self.best_loss)
        self.best_loss = min(loss, self.best_loss)
        if is_best:
            self.counter = 0
            self.save_weights(model, c, args)
            print('Weights saved.')
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
        else:
            return False
 
    def save_weights(self, model, c, args):
        """Save Deep SVDD weights."""
        torch.save({'model': model.state_dict(),
                    'center': c.cpu().data.numpy().tolist()}, 'deepsvdd/weights/model_parameters_{}.pth'.format(
                                                               args.anormal_class))