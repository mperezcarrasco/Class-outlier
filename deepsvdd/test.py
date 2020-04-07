import torch

from sklearn.metrics import roc_auc_score

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels1 = []
    labels2 = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y1, y2 in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())
            labels1.append(y1.cpu())
            labels2.append(y2.cpu())
    labels1, labels2, scores = torch.cat(labels1).numpy(), torch.cat(labels2).numpy(), torch.cat(scores).numpy()
    print('ROC AUC score: {:.3f}'.format(roc_auc_score(labels1, scores)))
    return labels1.reshape(-1,), labels2.reshape(-1,), scores