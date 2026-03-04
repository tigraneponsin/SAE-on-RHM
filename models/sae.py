import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=False)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

    def loss(self, x, lambda_l1=1e-3):
        recon, z = self(x)
        recon_loss = F.mse_loss(recon, x)
        sparse_loss = z.abs().mean()
        total_loss = recon_loss + lambda_l1 * sparse_loss
        return total_loss, recon_loss, sparse_loss

    @torch.no_grad()
    def activation_stats(self, z, threshold=1e-8):
        active_mask = z > threshold
        active_fraction = active_mask.float().mean().item()
        dead_features = (active_mask.sum(dim=0) == 0).sum().item()
        return {
            'active_fraction': active_fraction,
            'dead_features': dead_features
        }
