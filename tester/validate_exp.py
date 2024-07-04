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
from models import LatentModel
from losses import FlowConLoss

from einops import rearrange
from math import log, pi

def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def calc_likelihood(cfg, z, mu, log_sd, device, n_pixel, logdet):
  b, _ = z.size()
  z = rearrange(z, 'b d -> b 1 d')
  mu = rearrange(mu, 'b d -> 1 b d').repeat(b, 1, 1)
  log_sd = rearrange(log_sd, 'b d -> 1 b d').repeat(b, 1, 1)
  
  log_p_batch = gaussian_log_p(z, mu, log_sd)
  log_p_all = log_p_batch.sum(dim=(2))
  
  log_p_all = (-log(cfg.FLOW.N_BINS) * n_pixel) + log_p_all + logdet.mean()
  # ic(z.size(), mu.size(), log_sd.size(), log_p_batch.size(), log_p_all.size())
  return (log_p_all/ (log(2) * n_pixel))

def validate_flow(cfg, loader, pretrained, flow, mu, log_sd, device):
  n_pixel = cfg.FLOW.IN_FEAT
  y_pred = []
  y_true = []
  pretrained.eval()
  cls.eval()
  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)

    features = pretrained.intermediate_forward(x, cfg.TRAINING.PRT_LAYER)
    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)
    z, _, _, sdlj, _, _ = flow(features)
    log_probs = calc_likelihood(cfg, z, mu, log_sd, device, n_pixel, sdlj)
      
    y_pred += torch.argmax(log_probs, dim=-1).cpu().tolist()
    y_true += label.cpu().tolist()

  return y_pred, y_true

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
  model = LatentModel(cfg)
  model = model.to(device)

  flow_ckp = torch.load(f"{ckp_path}/{db}_{cfg.TRAINING.PRT_CONFIG}_{cfg.TRAINING.PRETRAINED}_layer{cfg.TRAINING.PRT_LAYER}_flow.pt", map_location=device)
  sd = {k: v for k, v in flow_ckp['state_dict'].items()}
  state = model.state_dict()
  state.update(sd)
  model.load_state_dict(state, strict=True)

  # load params 
  dist_dir = f"./data/distributions/{args.config}"
  dist_path = os.path.join(dist_dir, f"layer{cfg.TRAINING.PRT_LAYER}")
  mu = torch.load(os.path.join(dist_path, "mu.pt"), map_location=device)
  log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"), map_location=device)
  mu =  torch.stack(mu)
  log_sd =  torch.stack(log_sd)

  # LOADER
  id_transform = select_ood_transform(cfg, cfg.DATASET.IN_DIST, cfg.DATASET.IN_DIST)
  test_set = select_ood_testset(cfg, cfg.DATASET.IN_DIST, id_transform)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

  with torch.no_grad():
    y_pred, y_true = validate(cfg, test_loader, pretrained, cls, device)
    val_acc, val_err = get_metrics(y_true, y_pred)
    ic(val_acc)
    y_pred, y_true = validate_flow(cfg, test_loader, pretrained, model, mu, log_sd, device)
    val_acc, val_err = get_metrics(y_true, y_pred)
    ic(val_acc)