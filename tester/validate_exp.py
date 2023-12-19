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

from loaders import getTargetDataSet
from utils import seed_everything, get_args, get_metrics, mkdir
from configs import get_cfg_defaults
from models import LatentModel, DenseNet3, ResNet34, EfficientNet
from losses import FlowConLoss


def validate(cfg, loader, pretrained, device):
  y_pred = []
  y_true = []
  pretrained.eval()
  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)

    # _, _, features = pretrained.penultimate_forward(x)
    out = pretrained(x)
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

  # PRETRAINED MODEL
  if cfg.TRAINING.PRETRAINED == "densenet":
    pretrained = DenseNet3(100, cfg.DATASET.N_CLASS)
    pretrained = pretrained.to(device)
    checkpt = torch.load(f"{ckp_path}/{args.config}_densenet.pt", map_location=device)
  elif cfg.TRAINING.PRETRAINED == "resnet":
    pretrained = ResNet34(cfg.DATASET.N_CLASS)
    pretrained = pretrained.to(device)
    checkpt = torch.load(f"{ckp_path}/{db}_{cfg.TRAINING.PRT_CONFIG}_resnet.pt", map_location=device)
  elif cfg.TRAINING.PRETRAINED == "effnet":
    model = EfficientNet(cfg.DATASET.N_CLASS)


  sd = {k: v for k, v in checkpt['state_dict'].items()}
  state = pretrained.state_dict()
  state.update(sd)
  pretrained.load_state_dict(state, strict=True)
  
  transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

  
  # LOADER
  train_loader, test_loader = getTargetDataSet(cfg, cfg.DATASET.IN_DIST, cfg.TRAINING.BATCH, transform_train, transform_test, './data')

  with torch.no_grad():
    y_pred, y_true = validate(cfg, test_loader, pretrained, device)
  val_acc, val_err = get_metrics(y_true, y_pred)
  ic(val_acc)