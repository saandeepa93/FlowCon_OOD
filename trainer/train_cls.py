import sys
sys.path.append('.')
import torch 
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile

from imports import *

from utils import seed_everything, get_args, get_metrics, mkdir
from configs import get_cfg_defaults
from models import *
from loaders import select_dataset, select_classifier, select_transform

def validate(loader, model, cls, criterion, device):
  total_loss = []
  y_pred = []
  y_true = []
  model.eval()

  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)

    feats = model(x)
    out = F.softmax(cls(feats), dim=-1)
    loss = criterion(out, label)
    # loss = criterion(F.softmax(out, dim=-1), label)
    
    total_loss.append(loss.cpu().item())
    y_pred += torch.argmax(out, dim=-1).cpu().tolist()
    y_true += label.cpu().tolist()

  total_loss = sum(total_loss)/len(total_loss)
  return total_loss, y_pred, y_true

def train(loader, model, cls, criterion, optimizer, device):
  total_loss = []
  
  y_pred = []
  y_true = []

  model.train()
  for b, (x, label) in enumerate(loader, 0):
    x = x.to(device)
    label = label.to(device)
    optimizer.zero_grad()

    feats = model(x)
    out = F.softmax(cls(feats), dim=-1)
    loss = criterion(out, label)
    # loss = criterion(F.softmax(out, dim=-1), label)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
      total_loss.append(loss.cpu().item())
      y_pred += torch.argmax(out, dim=-1).cpu().tolist()
      y_true += label.cpu().tolist()

  total_loss = sum(total_loss)/len(total_loss)
  return total_loss, y_pred, y_true

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
  writer = SummaryWriter(f'./runs/{args.config}')

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.TRAINING.LR=1e-3
  cfg.freeze()
  print(cfg)

  # MODEL
  model, cls = select_classifier(cfg, cfg.DATASET.IN_DIST, cfg.TRAINING.PRETRAINED, cfg.DATASET.N_CLASS)
  train_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=True)
  test_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=False)
    

  model = model.to(device)
  cls = cls.to(device)
  print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  # LOSS, OPTIM, SCHEDULER
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(list(model.parameters()) + list(cls.parameters()), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)

  train_set = select_dataset(cfg, cfg.DATASET.IN_DIST, train_transform, train=True)
  if cfg.DATASET.IN_DIST in ['raf', 'aff'] and cfg.DATASET.W_SAMPLER:
    class_freq = np.array(list(train_set.cnt_dict.values()))
    weight = 1./class_freq
    sample_weight = torch.tensor([weight[t] for t in train_set.all_labels])
    sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight))
    train_loader = DataLoader(train_set, batch_size=cfg.TRAINING.BATCH, \
        num_workers=cfg.DATASET.NUM_WORKERS, sampler=sampler)
  else:
    train_loader = DataLoader(train_set, batch_size=cfg.TRAINING.BATCH, shuffle=True, num_workers = cfg.DATASET.NUM_WORKERS)

  test_set = select_dataset(cfg, cfg.DATASET.IN_DIST, test_transform, train=False)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)


  # TRAINING
  best_acc = 1e-6
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    train_loss, y_pred_train, y_true_train = train(train_loader, model, cls, criterion, optimizer, device)
    with torch.no_grad():
      val_loss, y_pred_val, y_true_val = validate(test_loader, model, cls, criterion, device)
    # scheduler.step()

    # COMPUTE METRICS
    train_acc, train_err = get_metrics(y_true_train, y_pred_train)
    val_acc, val_err = get_metrics(y_true_val, y_pred_val)

    # SAVE BEST MODEL
    if val_acc  > best_acc:
      best_acc = val_acc
      best_epoch = epoch
      ckp_dict = {
                'net_state_dict': model.state_dict(),
                'cls_state_dict': cls.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': cfg,
            }
      torch.save(ckp_dict, f"{ckp_path}/{args.config}_{cfg.TRAINING.PRETRAINED}.pt")

    pbar.set_description(
      f"Train Loss: {round(train_loss, 4)}; Val Loss: {round(val_loss, 4)}"
      f"Train Acc: {round(train_acc, 4)}; Val Acc: {round(val_acc, 4)}"
    )

    writer.add_scalar("Train/Loss", round(train_loss, 4), epoch)
    writer.add_scalar("Train/Acc", round(train_acc, 4), epoch)
    writer.add_scalar("Train/Error", round(train_err, 4), epoch)
    writer.add_scalar("Val/Loss", round(val_loss, 4), epoch)
    writer.add_scalar("Val/Acc", round(val_acc, 4), epoch)
    writer.add_scalar("Val/Error", round(val_err, 4), epoch)


