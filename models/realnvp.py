import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum

from icecream import ic 
from sys import exit as e



class iSequential(torch.nn.Sequential):

    def inverse(self, y):
        for module in reversed(self._modules.values()):
            assert hasattr(module,'inverse'), '{} has no inverse defined'.format(module)
            y = module.inverse(y)
        return y

    def logdet(self):
        log_det = 0
        for module in self._modules.values():
            assert hasattr(module,'logdet'), '{} has no logdet defined'.format(module)
            log_det += module.logdet()
        return log_det

    def reduce_func_singular_values(self,func):
        val = 0
        for module in self._modules.values():
            if hasattr(module,'reduce_func_singular_values'):
                val += module.reduce_func_singular_values(func)
        return val


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    TABULAR = 2
    HORIZONTAL = 3
    VERTICAL = 4
    Quadrant = 5
    SubQuadrant = 6
    Center = 7


class MaskTabular:
    def __init__(self, reverse_mask):
        self.type = MaskType.TABULAR
        self.reverse_mask = reverse_mask

    def mask(self, x):
        dim = x.size(1)
        split = dim // 2
        self.b = torch.zeros((1, dim), dtype=torch.float).to(x.device)
        
        if self.reverse_mask:
            self.b[:, split:] = 1.
            # x_id = x[:, split:]
            # x_change = x[:, :split]
        else:
            self.b[:, :split] = 1.
            # x_id = x[:, :split]
            # x_change = x[:, split: ]
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)
      
    def get_valid_half(self, x):
      dim = x.size(1)
      split = dim // 2
      if not self.reverse_mask:
          return x[:, :split]
      else:
        return x[:, split:]

class RescaleTabular(nn.Module):
    def __init__(self, D):
        super(RescaleTabular, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x

class CouplingLayerBase(nn.Module):
    """Coupling layer base class in RealNVP.
    
    must define self.mask, self.st_net, self.rescale
    """

    def _get_st(self, x):
        x_id, x_change = self.mask.mask(x)
        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))

        return s, t, x_id, x_change

    def forward(self, x, sldj=None, reverse=True):
        s, t, x_id, x_change = self._get_st(x)
        s, t = self.mask.mask_st_output(s, t)

        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s
        self._logdet = s.view(s.size(0), -1).sum(-1)
        if self.mask.type == MaskType.SubQuadrant:
            # DEBUG!!!!!!!
           self._logdet = self.mask.reshape_logdet(self._logdet) 
        x = self.mask.unmask(x_id, x_change)

        # LEARNED PRIOR
        x_change_valid = self.mask.get_valid_half(x_id)
        mean, log_sd = self.prior(x_change_valid).chunk(2, 1)

        return x, mean, log_sd, self._logdet

    def inverse(self, y):
        s, t, x_id, x_change = self._get_st(y)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = x_change * inv_exp_s - t
        self._logdet = -s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)

        return x

    def logdet(self):
        return self._logdet


class ZeroLinear(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
  
  def forward(self, input):
    out = self.linear(input)
    
    return out

class CouplingLayerTabular(CouplingLayerBase):

    def __init__(self, in_dim, mid_dim, num_layers, mask, init_zeros=False, dropout=False):
        
        super(CouplingLayerTabular, self).__init__()
        self.mask = mask
        self.st_net = nn.Sequential(nn.Linear(in_dim, mid_dim),
                                    nn.SiLU(),
                                    nn.Dropout(.5) if dropout else nn.Sequential(),
                                    *self._inner_seq(num_layers, mid_dim),
                                    ZeroLinear(mid_dim, in_dim*2)
                                    )
        self.prior = ZeroLinear(in_dim//2, in_dim*2)
        
        if init_zeros:
                # init w zeros to init a flow w identity mapping
                torch.nn.init.zeros_(self.st_net[-1].weight)
                torch.nn.init.zeros_(self.st_net[-1].bias)

        # self.rescale = nn.utils.weight_norm(RescaleTabular(in_dim))
        self.rescale = nn.utils.parametrizations.weight_norm(RescaleTabular(in_dim))

    @staticmethod
    def _inner_seq(num_layers, mid_dim):
        res = []
        for _ in range(num_layers):
            res.append(nn.Linear(mid_dim, mid_dim))
            res.append(nn.SiLU())
        return res

  
class RealNVPBase(nn.Module):

    def forward(self,x):
      # return self.body(x)
      logdet_all = 0
      for body in self.body:
        x, mean, log_sd, log_det = body(x)
        logdet_all += log_det
      return x, mean, log_sd, logdet_all

    def logdet(self):
        return self.body.logdet()

    def inverse(self, z):
        return self.body.inverse(z)

    def nll(self,x,y=None,label_weight=1.):
        z = self(x)
        logdet = self.logdet()
        z = z.reshape((z.shape[0], -1))
        prior_ll = self.prior.log_prob(z, y,label_weight=label_weight)
        nll = -(prior_ll + logdet)
        return nll

class RealNVPTabular(RealNVPBase):

    def __init__(self, in_dim=2, num_coupling_layers=6, hidden_dim=256, 
                 num_layers=2, init_zeros=False, dropout=False):

        super(RealNVPTabular, self).__init__()

        
        self.body = iSequential(*[
                        CouplingLayerTabular(
                            in_dim, hidden_dim, num_layers, MaskTabular(reverse_mask=bool(i%2)), init_zeros=init_zeros, dropout=dropout)
                        for i in range(num_coupling_layers)
                    ])
        