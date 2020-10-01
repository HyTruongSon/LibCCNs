import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

torch.manual_seed(123456789)

dtype_float = torch.float
dtype_int = torch.int
device = torch.device("cpu")

class VAE(nn.Module):
    """
    Variational Autoencoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graph):
        # Encoder
        encoded_vars = self.encoder(graph)
        N_channels = encoded_vars.shape[-1]
        assert(N_channels % 2 == 0), "Number of channels is not even!"
        z_var = encoded_vars[..., :N_channels//2]
        z_mu = encoded_vars[..., N_channels//2:]

        # Re-parameterization trick: Sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # Decoder
        predict = self.decoder(x_sample)

        return predict, z_mu, z_var

class AE(nn.Module):
    """
    Autoencoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graph):
        # Encoder
        encoded = self.encoder(graph)

        # Decoder
        predict = self.decoder(encoded)

        return predict