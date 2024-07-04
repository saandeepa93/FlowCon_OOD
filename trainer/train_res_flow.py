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
import wandb
from torch.profiler import profile, record_function, ProfilerActivity

from loaders import select_ood_testset, select_classifier, select_dataset, select_ood_transform, select_transform
from utils import *
from configs import get_cfg_defaults
from models import *
from losses import FlowConLoss


def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
  # T
  warm_epochs= cfg.LR.WARM_ITER
  warmup_from = cfg.LR.WARMUP_FROM
  warmup_to = cfg.TRAINING.LR
  if cfg.LR.WARM and epoch <= warm_epochs:
    p = (batch_id + (epoch - 1) * total_batches) / \
        (warm_epochs * total_batches)
    lr = warmup_from + p * (warmup_to - warmup_from)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(cfg, loader, epoch, pretrained, flow, criterion, optimizer, device):
  total_con_loss = []
  total_nll_loss = []
  flow.train()
  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)

    with torch.no_grad():
      # _, _, features = pretrained.penultimate_forward(x)
      features = pretrained.intermediate_forward(x, cfg.TRAINING.PRT_LAYER)
      features = features.view(features.size(0), features.size(1), -1)
      features = torch.mean(features, 2)
    
    # WARMUP LEARNING RATE
    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)
    
    z, means, log_sds, sldj, log_vars, logits = flow(features)
      # LOSS
    nll_loss, log_p, _, log_p_all = criterion.nllLoss(z, sldj, means, log_sds)
    log_p = log_p.mean()
    con_loss = criterion.conLoss(log_p_all, label)
    con_loss_mean = con_loss.mean()
    # loss = (1/cfg.FLOW.IN_FEAT * nll_loss) + con_loss_mean
    loss = (cfg.LOSS.LMBDA_MIN * nll_loss) + con_loss_mean
    # loss = nll_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      total_con_loss += con_loss.tolist()
      total_nll_loss.append(nll_loss.item())

  total_con_loss = sum(total_con_loss)/len(total_con_loss)
  total_nll_loss = sum(total_nll_loss)/len(total_nll_loss)
  
  return total_con_loss, total_nll_loss, log_p

def validate(cfg, loader, pretrained, flow, criterion, device):
  total_con_loss = []
  total_nll_loss = []
  flow.eval()
  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)

    # _, _, features = pretrained.penultimate_forward(x)
    features = pretrained.intermediate_forward(x, cfg.TRAINING.PRT_LAYER)
    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)

    z, means, log_sds, sldj, log_vars, logits = flow(features)
      # LOSS
    nll_loss, log_p, _, log_p_all = criterion.nllLoss(z, sldj, means, log_sds)
    log_p = log_p.mean()
    con_loss = criterion.conLoss(log_p_all, label)
    con_loss_mean = con_loss.mean()
    
    total_con_loss += con_loss.tolist()
    total_nll_loss.append(nll_loss.item())

  total_con_loss = sum(total_con_loss)/len(total_con_loss)
  total_nll_loss = sum(total_nll_loss)/len(total_nll_loss)
  
  return total_con_loss, total_nll_loss, log_p


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

  # SET TENSORBOARD PATH
  writer = wandb.init(project=f"{args.config}-flow")
  

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  # cfg.freeze()
  print(cfg)
  
  writer.config.update(cfg)
  
  pretrained, _ = select_classifier(cfg, cfg.DATASET.IN_DIST, cfg.TRAINING.PRETRAINED, cfg.DATASET.N_CLASS)

  # PRETRAINED MODEL
  if cfg.TRAINING.PRETRAINED in ["resnet18", 'resnet101', 'effnet']:
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg.DATASET.IN_DIST}_{cfg.TRAINING.PRETRAINED}.pt', map_location=device)
  elif cfg.TRAINING.PRETRAINED == "wideresnet":
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg.DATASET.IN_DIST}_{cfg.TRAINING.PRETRAINED}_40_2.pt', map_location=device)
  pretrained = pretrained.to(device)

  if cfg.DATASET.IN_DIST in ['raf', 'aff']:
    sd = {k: v for k, v in checkpoint['net_state_dict'].items()}
    state = pretrained.state_dict()
    state.update(sd)
    pretrained.load_state_dict(state, strict=True)
  else:
    pretrained.load_state_dict(checkpoint['net_state_dict'])

  for param in pretrained.parameters():
    param.requires_grad=False
  pretrained.eval()

  # FLOW MODEL
  model = LatentModel(cfg)
  model = model.to(device)
  print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

  # PREPARE OPTIMIZER
  optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  scheduler = CosineAnnealingLR(optimizer, cfg.LR.T_MAX, cfg.LR.MIN_LR)

  criterion = FlowConLoss(cfg, device, None)
  
  
  train_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=True)
  test_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=False)
  
  train_set = select_dataset(cfg, cfg.DATASET.IN_DIST, train_transform, train=True)
  train_loader = DataLoader(train_set, batch_size=cfg.TRAINING.BATCH, shuffle=True, num_workers = cfg.DATASET.NUM_WORKERS)

  test_set = select_dataset(cfg, cfg.DATASET.IN_DIST, test_transform, train=False)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

 
  best_loss = 1e6
  pbar = tqdm(range(cfg.TRAINING.ITER))
  # prof.start()
  for epoch in pbar:
    # Train and Validate
    train_con_loss, train_nll_loss, _ = train(cfg, train_loader, epoch, pretrained, model, criterion, optimizer, device)
    with torch.no_grad():
      val_con_loss, val_nll_loss, log_p = validate(cfg, test_loader, pretrained, model, criterion, device)
    # prof.step()

    # Save best
    if val_con_loss+val_nll_loss < best_loss:
      best_loss = val_con_loss+val_nll_loss
      best_epoch = epoch
      ckp_dict = {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': cfg,
                'best_epoch': best_epoch
            }
      torch.save(ckp_dict, f"{ckp_path}/{args.config}_{cfg.TRAINING.PRETRAINED}_layer{cfg.TRAINING.PRT_LAYER}_flow.pt")
    
    if epoch % 200 == 0:
      ckp_dict = {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': cfg,
                'best_epoch': best_epoch
            }
      epoch_ckp_dir = f"{ckp_path}/{args.config}"
      mkdir(epoch_ckp_dir)
      torch.save(ckp_dict, f"{epoch_ckp_dir}/{args.config}_{cfg.TRAINING.PRETRAINED}_layer{cfg.TRAINING.PRT_LAYER}_flow_{epoch}.pt")

    
    # Display Metrics
    pbar.set_description(
      f"Train Con Loss: {round(train_con_loss, 4)}; Train NLL Loss: {round(train_nll_loss, 4)};"\
      f"Val Con Loss: {round(val_con_loss, 4)}; Val NLL Loss: {round(val_nll_loss, 4)};"\
      f"Best Epoch: {best_epoch}; Log prob: {round(log_p.item(), 4)}; "\
    )

    # Log Metrics
    writer.log({
      "epoch": epoch, 
      "Train/NLL": round(train_nll_loss, 4), 
      "Train/Con": round(train_con_loss, 4),
      "Val/NLL": round(val_nll_loss, 4), 
      "Val/Con": round(val_con_loss, 4)
    })