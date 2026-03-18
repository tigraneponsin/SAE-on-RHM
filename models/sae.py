import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=True)
        self._reset_custom_parameters()

    def _reset_custom_parameters(self, decoder_col_norm=0.1, eps=1e-12):
        with torch.no_grad():
            # Decoder columns are random directions with fixed norm.
            w_dec = torch.randn(self.input_dim, self.latent_dim, device=self.decoder.weight.device)
            w_dec = w_dec / (w_dec.norm(dim=0, keepdim=True) + eps)
            w_dec = w_dec * float(decoder_col_norm)
            self.decoder.weight.copy_(w_dec)

            # Encoder starts as decoder transpose.
            self.encoder.weight.copy_(self.decoder.weight.t())

            # Biases start at zero for both encoder and decoder.
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    def forward(self, x):
        z = F.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z
    
    def decoder_feature_norms(self, eps=1e-12):
        w_dec = self.decoder.weight
        return torch.sqrt((w_dec ** 2).sum(dim=0) + eps)
    
    def loss(self, x, lambda_l1=5):
        recon, z = self(x)
        recon_loss = F.mse_loss(recon, x)
        #decoder weighted L1 penalty
        dec_norms = self.decoder_feature_norms()
        sparse_loss = (z.abs() * dec_norms.unsqueeze(0)).mean(dim=1).mean()
        total_loss = recon_loss + lambda_l1 * sparse_loss
        return total_loss, recon_loss, sparse_loss

    @torch.no_grad()
    def activation_stats(self, z, threshold=0):
        active_mask = z > threshold
        active_fraction = active_mask.float().mean().item()
        dead_features = (active_mask.sum(dim=0) == 0).sum().item()
        return {
            'active_fraction': active_fraction,
            'dead_features': dead_features
        }

   
