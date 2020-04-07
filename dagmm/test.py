import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

from dagmm.forward_step import ComputeLoss

def eval(model, dataloaders, device, n_gmm):
    """Testing the DAGMM model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, None, device, n_gmm)
    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
        for x, _, _ in dataloader_train:
            x = x.float().to(device)

            _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            
            N_samples += x.size(0)
            
        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
        

        # Obtaining Labels and energy scores for test data
        energy_test = []
        latent_test = []
        labels_test1 = []
        labels_test2 = []
        for x, y1, y2 in dataloader_test:
            x = x.float().to(device)

            _, _, z, gamma = model(x)
            sample_energy, cov_diag  = compute.compute_energy(z, gamma, train_phi,
                                                              train_mu, train_cov,
                                                              sample_mean=False)
            
            energy_test.append(sample_energy.detach().cpu())
            latent_test.append(z.detach().cpu())
            labels_test1.append(y1)
            labels_test2.append(y2)
        energy_test = torch.cat(energy_test).numpy()
        latent_test = torch.cat(latent_test).numpy()
        labels_test1 = torch.cat(labels_test1).numpy()
        labels_test2 = torch.cat(labels_test2).numpy()

    threshold = np.percentile(energy_test, 100 - 1)
    pred = (energy_test > threshold).astype(int)
    gt = labels_test1.astype(int)
    precision, recall, f_score, _ = prf(gt, pred, average='binary')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_test1, energy_test)*100))
    return labels_test1.reshape(-1,), labels_test2.reshape(-1,), energy_test, latent_test