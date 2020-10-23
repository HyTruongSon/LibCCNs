import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

torch.manual_seed(123456789)

dtype_float = torch.float
dtype_int = torch.int
device = torch.device("cpu")

class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder. '''
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # The encoder
        mu, sigma = self.enc(x)

        # Reparameterization trick
        std = torch.exp(sigma / 2)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)

        # The decoder
        predicted = self.dec(sample)
        return predicted, mu, sigma

class DGVAE(nn.Module):
    ''' This the DGVAE, which takes a encoder and decoder. '''
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # The encoder
        mu_0, log_sigma_0 = self.enc(x)

        # Reparameterization trick
        sigma_0 = torch.exp(log_sigma_0)
        eps = torch.randn_like(mu_0)
        sample = torch.softmax(mu_0 + torch.matmul(torch.diag(torch.sqrt(sigma_0)), eps), dim = 0)

        # The decoder
        predicted = self.dec(sample)
        return predicted, mu_0, sigma_0