import torch 
import timm
from torch import nn 
from .net import build_model


class Resnet101_Linear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.FLOW.IN_FEAT, cfg.DATASET.N_CLASS)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class Resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.res101 = build_model( 'ir_101')
        statedict = torch.load('./checkpoints/adaface_ir101_webface12m.ckpt')['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        self.res101.load_state_dict(model_statedict)

    def forward(self, x):
        feats, _ = self.res101(x)
        return feats
    
    def intermediate_forward(self, x, n_layer):
        feats, _ = self.res101(x)
        return feats


class EfficientNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.effnet =  timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    self.effnet.classifier = nn.Identity()
    self.effnet.load_state_dict(torch.load('./checkpoints/state_vggface2_enet0_new.pt', map_location=torch.device('cpu')))
  
  def forward(self, x):
    feats = self.effnet(x)
    return feats
    
  def intermediate_forward(self, x, n_layer):
    feats = self.effnet(x)
    return feats