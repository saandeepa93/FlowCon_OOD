import sys
sys.path.append('.')
from imports import *

import torch 
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile

from loaders import select_ood_testset, select_classifier, select_dataset, select_ood_transform, select_transform
from utils import seed_everything, get_args, get_metrics, mkdir
from configs import get_cfg_defaults
# from models import LatentModel, DenseNet3, ResNet34, EfficientNet
from losses import FlowConLoss


def validate(cfg, loader, pretrained, cls, device):
  y_pred = []
  y_true = []
  pretrained.eval()
  cls.eval()
  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)

    # _, _, features = pretrained.penultimate_forward(x)
    feats = pretrained(x)
    out = cls(feats)
    y_pred += torch.argmax(out, dim=-1).cpu().tolist()
    y_true += label.cpu().tolist()

  return y_pred, y_true


if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")
  ckp_path = os.path.join(f"./checkpoints/{db}")
  mkdir(ckp_path)

  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.TRAINING.LR = 1e-5
  cfg.TRAINING.ITER = 701
  cfg.TRAINING.BATCH = 64
  cfg.freeze()
  print(cfg)

  
  pretrained, cls = select_classifier(cfg, cfg.DATASET.IN_DIST, cfg.TRAINING.PRETRAINED, cfg.DATASET.N_CLASS)
  # PRETRAINED MODEL
  if cfg.TRAINING.PRETRAINED in ["resnet18", 'resnet101', 'effnet']:
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg.DATASET.IN_DIST}_{cfg.TRAINING.PRETRAINED}.pt', map_location=device)
  elif cfg.TRAINING.PRETRAINED == "wideresnet":
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg.DATASET.IN_DIST}_{cfg.TRAINING.PRETRAINED}_40_2.pt', map_location=device)

  if cfg.DATASET.IN_DIST in ['raf', 'aff']:
    sd = {k: v for k, v in checkpoint['net_state_dict'].items()}
    state = pretrained.state_dict()
    state.update(sd)
    pretrained.load_state_dict(state, strict=True)
  else:
    pretrained.load_state_dict(checkpoint['net_state_dict'])
    cls.load_state_dict(checkpoint['cls_state_dict'])
  
  pretrained = pretrained.to(device)
  cls = cls.to(device)

  # LOADER
  id_transform = select_ood_transform(cfg, cfg.DATASET.IN_DIST, cfg.DATASET.IN_DIST)
  test_set = select_ood_testset(cfg, cfg.DATASET.IN_DIST, id_transform)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

  with torch.no_grad():
    y_pred, y_true = validate(cfg, test_loader, pretrained, cls, device)
  val_acc, val_err = get_metrics(y_true, y_pred)
  ic(val_acc)