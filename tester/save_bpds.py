import sys
sys.path.append('.')
from imports import *

import torch 
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import DataLoader

from einops import rearrange
from math import log, pi

from loaders import select_ood_testset, select_classifier, select_dataset, select_ood_transform, select_transform
from utils import seed_everything, get_args, get_metrics, mkdir, make_roc, plot_umap, metric, get_metrics_ood
from configs import get_cfg_defaults
from models import *
from losses import FlowConLoss

import torch.profiler as profiler





def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def calc_likelihood(cfg1, z, mu, log_sd, device, n_pixel, logdet):
  b, _ = z.size()
  z = rearrange(z, 'b d -> b 1 d')
  mu = rearrange(mu, 'b d -> 1 b d').repeat(b, 1, 1)
  log_sd = rearrange(log_sd, 'b d -> 1 b d').repeat(b, 1, 1)
  
  log_p_batch = gaussian_log_p(z, mu, log_sd)
  log_p_all = log_p_batch.sum(dim=(2))
  
  log_p_all = (-log(cfg1.FLOW.N_BINS) * n_pixel) + log_p_all + logdet.mean()
  # ic(z.size(), mu.size(), log_sd.size(), log_p_batch.size(), log_p_all.size())
  return (log_p_all/ (log(2) * n_pixel))

def calc_emp_params(cfg1, args, loader, pretrained, flow, dist_dir, device, labels_in_ood=None):
  z_all, mu_all, log_sd_all = [], [], []
  labels_all = []

  pretrained.eval()
  flow.eval()
  for b, (x, label) in enumerate(tqdm(loader), 0):
    x = x.to(device)
    label = label.to(device)

    # _, _, features = pretrained.penultimate_forward(x)
    features = pretrained.intermediate_forward(x, cfg1.TRAINING.PRT_LAYER)

    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)
    z, means, log_sds, sldj, log_vars, logits = flow(features)

    z_all.append(z)
    mu_all.append(means)
    log_sd_all.append(log_sds)

    labels_all.append(label)

  labels_all = torch.cat(labels_all, dim=0)
  z = torch.cat(z_all, dim=0)

  # plot_umap(cfg1, z.cpu(), labels_all.cpu(), f"{args.config}", 2, "in_ood", labels_in_ood)

  mu_k, std_k , z_k= [], [], []
  for cls in range(cfg1.DATASET.N_CLASS):
    cls_indices = (labels_all == cls).nonzero().squeeze()
    mu_k.append(torch.index_select(torch.cat(mu_all, dim=0), 0, cls_indices).mean(0))
    std_k.append(torch.index_select(torch.cat(log_sd_all, dim=0), 0, cls_indices).mean(0))
    # z_k.append(torch.index_select(z, 0, cls_indices).mean(0))
  
  torch.save(mu_k, os.path.join(dist_dir, "mu.pt"))
  torch.save(std_k, os.path.join(dist_dir, "log_sd.pt"))


def calc_scores_2(cfg1, args, loader, pretrained, flow, cls, mu, log_sd, criterion, loss_criterion, device, flg=False):
  n_pixel = cfg1.FLOW.IN_FEAT
  scores_all = []

  pretrained.eval()
  flow.eval()
  cls.eval()
  for b, (x, _) in enumerate(tqdm(loader), 0):
    x = x.to(device)
    x= Variable(x, requires_grad = True)
    
    # _, _, features = pretrained.penultimate_forward(x)
    logits = cls(pretrained(x))
    label = torch.argmax(logits, dim=-1)
    label = label.type(torch.int64)


    features = pretrained.intermediate_forward(x, cfg1.TRAINING.PRT_LAYER)
    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)
    
    z, _, _, sdlj, _, _ = flow(features)
    log_probs = calc_likelihood(cfg1, z, mu, log_sd, device, n_pixel, sdlj)
    loss = loss_criterion(log_probs, label)
    
    # loss = torch.mean(-torch.max(log_probs, dim=-1)[0])
    # loss = loss_criterion(F.softmax(log_probs, dim=-1), label)

    loss.backward()

    # INPUT PREPROCESSING
    # NEG. FGSM METHOD
    gradient =  torch.ge(x.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    with torch.no_grad():
      if cfg1.TRAINING.PRETRAINED == 'wideresnet':
          gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.5))
      elif cfg1.TRAINING.PRETRAINED == 'resnet18':
          gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.5))
      noisy_x = torch.add(x.data, -cfg1.TEST.MAGNITUDE, gradient)
      
      # _, _, features = pretrained.penultimate_forward(noisy_x)
      features = pretrained.intermediate_forward(noisy_x, cfg1.TRAINING.PRT_LAYER)
      features = features.view(features.size(0), features.size(1), -1)
      features = torch.mean(features, 2)

      # z,n_mu, n_log_sd, sdlj, _, _ = flow(features)
      # log_probs = calc_likelihood(cfg1, z, n_mu, n_log_sd, device, n_pixel, sdlj)
      # score = torch.max(log_probs, dim=-1)[0].cpu().tolist()
      
      z, n_mu, n_log_sd, sdlj, _, _ = flow(features)
      # _, score, _, _ = criterion.nllLoss(z, sdlj, n_mu, n_log_sd) # CIFAR10_4_WIDENET
      score, _, _, _ = criterion.nllLoss(z, sdlj, n_mu, n_log_sd) #CIFAR10_3_RESNET
      score = score.detach().cpu().tolist()



      scores_all += score
      # if flg:
      #   ic(score)
      #   e()

  return np.array(scores_all)




if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  
  db1 = args.config.split("_")[0]
  config_path1 = os.path.join(f"./configs/experiments/{db1}", f"{args.config}.yaml")
  # ckp_path = os.path.join(f"./checkpoints/{db1}")
  ckp_path = os.path.join(f"./checkpoints/{db1}/{args.config}")
  
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  # LOAD CONFIGURATION
  cfg1 = get_cfg_defaults()
  cfg1.merge_from_file(config_path1)
  cfg1.TRAINING.BATCH=192
  print(cfg1)
  
  pretrained, cls = select_classifier(cfg1, cfg1.DATASET.IN_DIST, cfg1.TRAINING.PRETRAINED, cfg1.DATASET.N_CLASS)

  # PRETRAINED MODEL
  if cfg1.TRAINING.PRETRAINED in ["resnet18", 'resnet101', 'effnet']:
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg1.DATASET.IN_DIST}_{cfg1.TRAINING.PRETRAINED}.pt', map_location=device)

  elif cfg1.TRAINING.PRETRAINED == "wideresnet":
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg1.DATASET.IN_DIST}_{cfg1.TRAINING.PRETRAINED}_40_2.pt', map_location=device)

  if cfg1.DATASET.IN_DIST in ['raf', 'aff']:
    sd = {k: v for k, v in checkpoint['net_state_dict'].items()}
    state = pretrained.state_dict()
    state.update(sd)
    pretrained.load_state_dict(state, strict=True)
  else:
    pretrained.load_state_dict(checkpoint['net_state_dict'])
  cls.to(device)
  cls.load_state_dict(checkpoint['cls_state_dict'])
  pretrained = pretrained.to(device)
  pretrained.load_state_dict(checkpoint['net_state_dict'])


  # FLOW MODEL
  model = LatentModel(cfg1)
  model = model.to(device)
  print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

  # id_transform = select_ood_transform(cfg1, cfg1.DATASET.IN_DIST, cfg1.DATASET.IN_DIST)
  # test_set = select_ood_testset(cfg1, cfg1.DATASET.IN_DIST, id_transform)
  id_transform = select_transform(cfg1, cfg1.DATASET.IN_DIST, pretrain=False)
  ic(id_transform)
  test_set = select_dataset(cfg1, cfg1.DATASET.IN_DIST, id_transform, train=False)
  train_set = select_dataset(cfg1, cfg1.DATASET.IN_DIST, id_transform, train=True)
  train_loader = DataLoader(train_set, batch_size=cfg1.TRAINING.BATCH, shuffle=False, num_workers = cfg1.DATASET.NUM_WORKERS)
  test_loader = DataLoader(test_set, batch_size=cfg1.TRAINING.BATCH, shuffle=False, num_workers = cfg1.DATASET.NUM_WORKERS)
  
  
  criterion = FlowConLoss(cfg1, device)
  loss_criterion = nn.CrossEntropyLoss()

  scores_in_layer = []
  scores_ood_layer = []
  avg_result_dict = {"rocauc": 0, "aupr_success": 0, "aupr_error": 0, "fpr": 0}
  model = LatentModel(cfg1)
  model = model.to(device)

  # FLOW MODEL
  # flow_ckp = torch.load(f"{ckp_path}/{db1}_{cfg1.TRAINING.PRT_CONFIG}_{cfg1.TRAINING.PRETRAINED}_layer{cfg1.TRAINING.PRT_LAYER}_flow.pt", map_location=device)
  flow_ckp = torch.load(f"{ckp_path}/{db1}_{cfg1.TRAINING.PRT_CONFIG}_{cfg1.TRAINING.PRETRAINED}_layer{cfg1.TRAINING.PRT_LAYER}_flow_600.pt", map_location=device)
  sd = {k: v for k, v in flow_ckp['state_dict'].items()}
  state = model.state_dict()
  state.update(sd)
  model.load_state_dict(state, strict=True)
  print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

  dist_dir = f"./data/distributions/{args.config}"
  mkdir(dist_dir)
  dist_path = os.path.join(dist_dir, f"layer{cfg1.TRAINING.PRT_LAYER}")
  mkdir(dist_path)

  
  # if cfg1.TEST.EMP_PARAMS:
  #   print("Saving params...")
  #   with torch.no_grad():
  #     calc_emp_params(cfg1, args, train_loader, pretrained, model, dist_path, device)
  
  # load params
  mu = torch.load(os.path.join(dist_path, "mu.pt"), map_location=device)
  log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"), map_location=device)
  mu =  torch.stack(mu)
  log_sd =  torch.stack(log_sd)
  
  # bpds_train = calc_scores_2(cfg1, args, train_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
  bpds_test = calc_scores_2(cfg1, args, test_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
  save_path = os.path.join("./data/bpds", args.config)
  
  mkdir(save_path)
  # np.save(f"{save_path}/{args.config}_train.npy", bpds_train)
  np.save(f"{save_path}/{args.config}_test", bpds_test)
  
  ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365', 'raf']
  # ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
  # ood_datasets = ['raf']
  for ood_ds in ood_datasets:
    if ood_ds == db1:
      continue
    ood_transform = select_ood_transform(cfg1, ood_ds, cfg1.DATASET.IN_DIST)
    ood_dataset = select_ood_testset(cfg1, ood_ds, ood_transform)
    ood_loader = DataLoader(ood_dataset, batch_size=cfg1.TRAINING.BATCH, shuffle=False, num_workers = cfg1.DATASET.NUM_WORKERS)
    scores_out = calc_scores_2(cfg1, args, ood_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
    # scores_out = scores_out[:num_ood]
    scores_out = np.array(scores_out)
    scores_out = scores_out[~np.isnan(scores_out)]
    scores_out = scores_out[np.isfinite(scores_out)]
    
    np.save(f"{save_path}/{ood_ds}_test", scores_out)
  

  
    

