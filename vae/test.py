import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

def eval(net, dataloader, device):
    """Testing the VAE model"""

    scores = []
    latents = []
    labels1 = []
    labels2 = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y1, y2 in dataloader:
            x = x.float().to(device)
            x_hat, _, _, z = net(x)
            score = F.mse_loss(x_hat, x, reduction='none')
            score = torch.sum(score, dim=(1,2,3))

            scores.append(score.detach().cpu())
            latents.append(z.detach().cpu())
            labels1.append(y1.cpu())
            labels2.append(y2.cpu())
            
    labels1, labels2 = torch.cat(labels1).numpy(), torch.cat(labels2).numpy()
    scores, latents = torch.cat(scores).numpy(), torch.cat(latents).numpy()
    print('ROC AUC score: {:.3f}'.format(roc_auc_score(labels2, scores)))
    return labels1, labels2, scores, latents