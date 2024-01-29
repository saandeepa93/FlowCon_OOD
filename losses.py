import torch 
from torch import nn 

from math import log, pi, exp


def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


class FlowConLoss:
  def __init__(self, cfg, device, p_y=None):
    self.cfg = cfg
    self.device=device
    self.n_bins = cfg.FLOW.N_BINS
    self.device = device
    self.n_pixel = cfg.FLOW.IN_FEAT

    self.tau = cfg.LOSS.TAU
    self.tau2 = cfg.LOSS.TAU2


    # RAF12
    self.init_loss = -log(self.n_bins) * self.n_pixel


  def nllLoss(self, z, logdet, mu, log_sd):
    b_size, _ = z.size()

    # Calculate total log_p
    log_p_total = 0
    log_p_all = torch.zeros((b_size, b_size), dtype=torch.float32, device=self.device)

    # Create mask to select NLL loss elements
    b, d = z.size()
    z = z.view(b, 1, d)

    nll_mask = torch.eye(b, device=self.device).view(b, b, 1)
    nll_mask = nll_mask.repeat(1, 1, d)

    # Square matrix for contrastive loss evaluation      
    log_p_batch = gaussian_log_p(z, mu, log_sd)

    # NLL losses
    log_p_nll = (log_p_batch * nll_mask).sum(dim=(2))
    log_p_nll = log_p_nll.sum(dim=1)

    log_p_all += log_p_batch.sum(dim=(2))

    logdet = logdet.mean()
    loss = self.init_loss + logdet + log_p_nll
    
    return ( 
      (-loss / (log(2) * self.n_pixel)).mean(), # CONVERTING LOGe to LOG2 |
      (log_p_nll / (log(2) * self.n_pixel)).mean(), #                     v
      # (loss / (log(2) * self.n_pixel)), # CONVERTING LOGe to LOG2 |
      # (log_p_nll / (log(2) * self.n_pixel)), #                     v
      (logdet / (log(2) * self.n_pixel)).mean(), 
      (log_p_all/ (log(2) * self.n_pixel))
  )


  def conLoss(self, log_p_all, labels):
    b, _ = log_p_all.size()
    
    # Create similarity and dissimilarity masks
    off_diagonal = torch.ones((b, b), device=self.device) - torch.eye(b, device=self.device)
    
    # Create label clone
    labels_orig = labels.clone()
    labels = labels.contiguous().view(-1, 1)

    # Create similarity masks
    sim_mask = torch.eq(labels, labels.T).float().to(self.device) * off_diagonal

    # Get respective log Probablities to compute row-wise pairwise against b*b log_p_all matrix
    # p_new_y = torch.index_select(self.p_y, 0, labels_orig)
    diag_logits = (log_p_all * torch.eye(b).to(self.device)).sum(dim=-1)

    # Compute pairwise bhatta coeff. (0.5* (8, 8) + (8, 1))
    pairwise = (self.tau * (log_p_all.contiguous().view(b, b) + diag_logits.view(b, 1)))
    pairwise_exp = torch.div(torch.exp(
      pairwise - torch.max(pairwise, dim=1, keepdim=True)[0]) + 1e-5, self.tau2)

    # Division term    
    pos_count = sim_mask.sum(1)
    pos_count[pos_count == 0] = 1

    # LOG PROB
    log_prob = pairwise_exp - (pairwise_exp.exp() * off_diagonal).sum(-1, keepdim=True).log()

    # compute mean against positive classes
    mean_log_prob_pos = (sim_mask * log_prob).sum(1) / pos_count
    
    return -mean_log_prob_pos