
import torch 
from torch import nn
from torch.nn import functional as F
from .realnvp import RealNVPTabular





class LatentModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg
    # BACKBONE
    self.flow = RealNVPTabular(in_dim=cfg.FLOW.IN_FEAT, hidden_dim=cfg.FLOW.MLP_DIM, num_layers=cfg.FLOW.N_FLOW, \
                    num_coupling_layers=cfg.FLOW.N_BLOCK, init_zeros=cfg.FLOW.INIT_ZEROS, dropout=cfg.TRAINING.DROPOUT)
    
    self.sigma1 = nn.Parameter(torch.zeros(1))
    self.sigma2 = nn.Parameter(torch.zeros(1))
    # self.sigma3 = nn.Parameter(torch.zeros(1))

    self.decode = nn.Linear(cfg.FLOW.IN_FEAT, cfg.DATASET.N_CLASS)
    # self.decode = nn.Linear(cfg.FLOW.IN_FEAT, cfg.FLOW.IN_FEAT)


  def forward(self, x):
    x, mean, log_sd, logdet = self.flow(x)
    z_repar = self.reparameterize(mean, log_sd)
    logits = self.decode(z_repar)
    return x, mean, log_sd, logdet, [self.sigma1,self.sigma2], F.softmax(logits, dim=-1)
    # return x, mean, log_sd, logdet, [self.sigma1,self.sigma2], F.softmax(logits, dim=-1)
    # return x, mean, log_sd, logdet, [self.sigma1,self.sigma2, self.sigma3], F.normalize(z_repar, dim=-1)


  def reparameterize(self, mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu