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




def validate_results(cfg, args, epoch, model, pretrained, device, criterion, train_loader, test_loader):
    # PRETRAINED MODEL
  if cfg.TRAINING.PRETRAINED == "resnet18":
    cls = ResNet_Linear(cfg.DATASET.N_CLASS)
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg.DATASET.IN_DIST}_{cfg.TRAINING.PRETRAINED}.pt', map_location=device)

  elif cfg.TRAINING.PRETRAINED == "wideresnet":
    cls = WideResNet_Linear(2, num_classes=cfg.DATASET.N_CLASS)
    checkpoint = torch.load(f'./checkpoints/classifiers/{cfg.DATASET.IN_DIST}_{cfg.TRAINING.PRETRAINED}_40_2.pt', map_location=device)

  cls.to(device)
  cls.load_state_dict(checkpoint['cls_state_dict'])

  loss_criterion = nn.CrossEntropyLoss()

  result = {}
  ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
  # magnitude = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
  # magnitude = list(np.linspace(0.0024, 0.005, 50, dtype=np.float32))
  magnitude = [cfg.TEST.MAGNITUDE]
  for ctr2, mag in enumerate(magnitude):
    ic(mag)
    cfg.TEST.MAGNITUDE = float(mag)
    result[str(mag)] = []
    
    scores_in_layer = []
    scores_ood_layer = []
    avg_result_dict = {"rocauc": 0, "aupr_success": 0, "aupr_error": 0, "fpr": 0}
    layers = [cfg.TRAINING.PRT_LAYER]
    for ctr, layer in enumerate(layers):
      cfg.FLOW.IN_FEAT = cfg.TEST.IN_FEATS[layer-1]
      cfg.TRAINING.PRT_LAYER = layer

      dist_dir = f"./data/distributions/{args.config}"
      mkdir(dist_dir)
      dist_path = os.path.join(dist_dir, f"layer{cfg.TRAINING.PRT_LAYER}")
      mkdir(dist_path)

      
      if cfg.TEST.EMP_PARAMS:
        if ctr == 0 and ctr2==0:
          print("Saving params...")
          with torch.no_grad():
            calc_emp_params(cfg, args, train_loader, pretrained, model, dist_path, device)
          
      
      # load params
      mu = torch.load(os.path.join(dist_path, "mu.pt"), map_location=device)
      log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"), map_location=device)
      mu =  torch.stack(mu)
      log_sd =  torch.stack(log_sd)
      scores_in = calc_scores_2(cfg, args, test_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
      with torch.no_grad():
        validate(cfg, args, test_loader, pretrained, model, mu, log_sd, device)

      scores_in_ds = []
      scores_ood_ds = []
      for ood_ds in ood_datasets:
        if cfg.DATASET.IN_DIST == ood_ds:
          continue
        print(f"Running {ood_ds}")
        if cfg.TEST.SCORE:
          ood_transform = select_ood_transform(ood_ds, cfg.DATASET.IN_DIST)
          ood_dataset = select_ood_testset(ood_ds, ood_transform)
          ood_loader = DataLoader(ood_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)
          
          print("Calculating Scores...")
          scores_out = calc_scores_2(cfg, args, ood_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
          scores_out = scores_out[:num_ood]

          scores = np.concatenate([np.array(scores_in), np.array(scores_out)], axis=0)
          labels = np.concatenate([np.ones(len(test_set)), np.zeros(num_ood)], axis=0)
          results_dict = get_metrics_ood(labels, scores, invert_score=False)

          avg_result_dict["rocauc"] += results_dict["rocauc"]
          avg_result_dict["aupr_success"] += results_dict["aupr_success"]
          avg_result_dict["aupr_error"] += results_dict["aupr_error"]
          avg_result_dict["fpr"] += results_dict["fpr"]

          # result[str(mag)].append(metric(np.array(scores_in), np.array(scores_out), stype=ood_ds))
          ic(results_dict)
      avg_result_dict["rocauc"] = avg_result_dict["rocauc"] / len(ood_datasets)
      avg_result_dict["aupr_success"] = avg_result_dict["aupr_success"] / len(ood_datasets)
      avg_result_dict["aupr_error"] = avg_result_dict["aupr_error"] / len(ood_datasets)
      avg_result_dict["fpr"] = avg_result_dict["fpr"] / len(ood_datasets)
      
      ic(avg_result_dict)
      result[str(mag)].append(avg_result_dict)
  res_path = os.path.join("./data/results", f"{db}_{cfg.TRAINING.PRT_CONFIG}")
  mkdir(res_path)
  with open(f'{res_path}/{epoch}_avg_{cfg.TRAINING.PRETRAINED}_{cfg.TRAINING.PRT_LAYER}_{cfg.TEST.MAGNITUDE}.json', 'w') as fp:
    json.dump(avg_result_dict, fp, indent=4)
  with open(f'{res_path}/{epoch}_{cfg.TRAINING.PRETRAINED}_{cfg.TRAINING.PRT_LAYER}_{cfg.TEST.MAGNITUDE}.json', 'w') as fp:
    json.dump(result, fp, indent=4)

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
  return (log_p_all/ (log(2) * n_pixel))

def calc_emp_params(cfg, args, loader, pretrained, flow, dist_dir, device, labels_in_ood=None):
  z_all, mu_all, log_sd_all = [], [], []
  labels_all = []

  pretrained.eval()
  flow.eval()
  for b, (x, label) in enumerate(tqdm(loader), 0):
    x = x.to(device)
    label = label.to(device)

    # _, _, features = pretrained.penultimate_forward(x)
    features = pretrained.intermediate_forward(x, cfg.TRAINING.PRT_LAYER)

    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)
    z, means, log_sds, sldj, log_vars, logits = flow(features)

    z_all.append(z)
    mu_all.append(means)
    log_sd_all.append(log_sds)

    labels_all.append(label)

  labels_all = torch.cat(labels_all, dim=0)
  z = torch.cat(z_all, dim=0)

  # plot_umap(cfg, z.cpu(), labels_all.cpu(), f"{args.config}", 2, "in_ood", labels_in_ood)

  mu_k, std_k , z_k= [], [], []
  for cls in range(cfg.DATASET.N_CLASS):
    cls_indices = (labels_all == cls).nonzero().squeeze()
    mu_k.append(torch.index_select(torch.cat(mu_all, dim=0), 0, cls_indices).mean(0))
    std_k.append(torch.index_select(torch.cat(log_sd_all, dim=0), 0, cls_indices).mean(0))
    # z_k.append(torch.index_select(z, 0, cls_indices).mean(0))
  
  torch.save(mu_k, os.path.join(dist_dir, "mu.pt"))
  torch.save(std_k, os.path.join(dist_dir, "log_sd.pt"))

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def calc_scores_2(cfg, args, loader, pretrained, flow, cls, mu, log_sd, criterion, loss_criterion, device, flg=False):
  n_pixel = cfg.FLOW.IN_FEAT
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

    features = pretrained.intermediate_forward(x, cfg.TRAINING.PRT_LAYER)
    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)
    
    z, _, _, sdlj, _, _ = flow(features)
    log_probs = calc_likelihood(cfg, z, mu, log_sd, device, n_pixel, sdlj)
    # loss = torch.mean(-torch.max(log_probs, dim=-1)[0])
    # loss = loss_criterion(F.softmax(log_probs, dim=-1), label)
    loss = loss_criterion(log_probs, label)

    loss.backward()

    # INPUT PREPROCESSING
    # NEG. FGSM METHOD
    gradient =  torch.ge(x.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    with torch.no_grad():
      if cfg.TRAINING.PRETRAINED == 'wideresnet':
          # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
          # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
          # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
          gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.5))
      elif cfg.TRAINING.PRETRAINED == 'resnet18':
      #     gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
      #     gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
      #     gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
          gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.5))
      noisy_x = torch.add(x.data, -cfg.TEST.MAGNITUDE, gradient)
      
      # _, _, features = pretrained.penultimate_forward(noisy_x)
      features = pretrained.intermediate_forward(noisy_x, cfg.TRAINING.PRT_LAYER)
      features = features.view(features.size(0), features.size(1), -1)
      features = torch.mean(features, 2)

      # z, _, _, sdlj, _, _ = flow(features)
      # log_probs = calc_likelihood(cfg, z, mu, log_sd, device, n_pixel, sdlj)
      # score = torch.max(log_probs, dim=-1)[0].cpu().tolist()
      
      z, mu, log_sd, sdlj, _, _ = flow(features)
      # _, score, _, _ = criterion.nllLoss(z, sdlj, mu, log_sd) # CIFAR10_4_WIDENET
      score, _, _, _ = criterion.nllLoss(z, sdlj, mu, log_sd) #CIFAR10_RESNET
      score = score.detach().cpu().tolist()

      scores_all += score
      if flg:
        ic(score)
        e()

  return scores_all

def validate(cfg, args, loader, pretrained, flow, mu, log_sd, device):
  n_pixel = cfg.FLOW.IN_FEAT
  y_val_pred = []
  y_val_true = []

  pretrained.eval()
  flow.eval()
  for b, (x, label) in enumerate(tqdm(loader), 0):
    x = x.to(device)
    label = label.to(device)
    features = pretrained.intermediate_forward(x, cfg.TRAINING.PRT_LAYER)
    features = features.view(features.size(0), features.size(1), -1)
    features = torch.mean(features, 2)
    
    z, _, _, sdlj, _, _ = flow(features)
    log_probs = calc_likelihood(cfg, z, mu, log_sd, device, n_pixel, sdlj)
    
    pred = torch.argmax(log_probs, dim=1)
    y_val_pred += pred.cpu().tolist()
    y_val_true += label.cpu().tolist()
    
  val_acc, _ = get_metrics(y_val_true, y_val_pred)
  ic(val_acc)

def calc_score(cfg, args, loader, pretrained, flow, mu, log_sd, criterion, device):
  n_pixel = cfg.FLOW.IN_FEAT
  scores_all = []

  pretrained.eval()
  flow.eval()
  for b, (x, label) in enumerate(tqdm(loader), 0):
    x = x.to(device)
    label = label.to(device)
    _, _, features = pretrained.penultimate_forward(x)
    z, mu, log_sd, sdlj, _, _ = flow(features)
    
    nll_loss, score, _, log_p_all = criterion.nllLoss(z, sdlj, mu, log_sd)
    scores_all += score.tolist()

    log_probs = calc_likelihood(cfg, z, mu, log_sd, device, n_pixel, sdlj)
    score = torch.max(log_probs, dim=-1)[0].tolist()


  return scores_all


if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")
  ckp_path = os.path.join(f"./checkpoints/{db}")
  # ckp_path = os.path.join(f"./checkpoints/main")
  mkdir(ckp_path)

  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.TRAINING.BATCH=64
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
  cls.to(device)
  cls.load_state_dict(checkpoint['cls_state_dict'])
  pretrained = pretrained.to(device)
  pretrained.load_state_dict(checkpoint['net_state_dict'])


  # FLOW MODEL
  model = LatentModel(cfg)
  model = model.to(device)
  print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

  id_transform = select_ood_transform(cfg, cfg.DATASET.IN_DIST, cfg.DATASET.IN_DIST)
  test_set = select_ood_testset(cfg, cfg.DATASET.IN_DIST, id_transform)
  num_ood = len(test_set) // 5
  train_set = select_dataset(cfg, cfg.DATASET.IN_DIST, id_transform, train=True)

  train_loader = DataLoader(train_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)


  ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
  # ood_datasets = ['aff']
  result = {}

  criterion = FlowConLoss(cfg, device)


  loss_criterion = nn.CrossEntropyLoss()

  # magnitude = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
  # magnitude = list(np.linspace(0.0024, 0.005, 50, dtype=np.float32))
  magnitude = [cfg.TEST.MAGNITUDE]
  for ctr2, mag in enumerate(magnitude):
    ic(mag)
    cfg.TEST.MAGNITUDE = float(mag)
    result[str(mag)] = []
    
    scores_in_layer = []
    scores_ood_layer = []
    avg_result_dict = {"rocauc": 0, "aupr_success": 0, "aupr_error": 0, "fpr": 0}
    layers = [cfg.TRAINING.PRT_LAYER]
    for ctr, layer in enumerate(layers):
      cfg.FLOW.IN_FEAT = cfg.TEST.IN_FEATS[layer-1]
      cfg.TRAINING.PRT_LAYER = layer
      model = LatentModel(cfg)
      model = model.to(device)

      # FLOW MODEL
      flow_ckp = torch.load(f"{ckp_path}/{db}_{cfg.TRAINING.PRT_CONFIG}_{cfg.TRAINING.PRETRAINED}_layer{cfg.TRAINING.PRT_LAYER}_flow.pt", map_location=device)
      sd = {k: v for k, v in flow_ckp['state_dict'].items()}
      state = model.state_dict()
      state.update(sd)
      model.load_state_dict(state, strict=True)
      print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

      dist_dir = f"./data/distributions/{args.config}"
      mkdir(dist_dir)
      dist_path = os.path.join(dist_dir, f"layer{cfg.TRAINING.PRT_LAYER}")
      mkdir(dist_path)

      
      if cfg.TEST.EMP_PARAMS:
        if ctr == 0 and ctr2==0:
          print("Saving params...")
          # with torch.no_grad():
            # calc_emp_params(cfg, args, train_loader, pretrained, model, dist_path, device)
      
      # load params
      mu = torch.load(os.path.join(dist_path, "mu.pt"), map_location=device)
      log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"), map_location=device)
      mu =  torch.stack(mu)
      log_sd =  torch.stack(log_sd)
      scores_in = calc_scores_2(cfg, args, test_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
      # with torch.no_grad():
      #   validate(cfg, args, test_loader, pretrained, model, mu, log_sd, device)

      scores_in_ds = []
      scores_ood_ds = []
      for ood_ds in ood_datasets:
        if cfg.DATASET.IN_DIST == ood_ds:
          continue
        print(f"Running {ood_ds}")
        if cfg.TEST.SCORE:
          ood_transform = select_ood_transform(cfg, ood_ds, cfg.DATASET.IN_DIST)
          ood_dataset = select_ood_testset(cfg, ood_ds, ood_transform)
          ood_loader = DataLoader(ood_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

          # UMAP EMBEDDING
          # with torch.no_grad():
          #   in_ood_sets = torch.utils.data.ConcatDataset([test_set, ood_dataset])
          #   in_ood_loader = DataLoader(in_ood_sets, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)
          #   lables = torch.cat([torch.ones(len(test_set), dtype=torch.int8), torch.zeros(len(ood_dataset), dtype=torch.int8)], dim=0)
          #   calc_emp_params(cfg, args, in_ood_loader, pretrained, model, dist_path, device, lables)
          #   e()
          
          print("Calculating Scores...")
          scores_out = calc_scores_2(cfg, args, ood_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
          scores_out = scores_out[:num_ood]

          scores = np.concatenate([np.array(scores_in), np.array(scores_out)], axis=0)
          labels = np.concatenate([np.ones(len(test_set)), np.zeros(num_ood)], axis=0)
          ic(len(scores_in), len(test_set),len(scores_out), num_ood )
          results_dict = get_metrics_ood(labels, scores, invert_score=False)

          avg_result_dict["rocauc"] += results_dict["rocauc"]
          avg_result_dict["aupr_success"] += results_dict["aupr_success"]
          avg_result_dict["aupr_error"] += results_dict["aupr_error"]
          avg_result_dict["fpr"] += results_dict["fpr"]

          # result[str(mag)].append(metric(np.array(scores_in), np.array(scores_out), stype=ood_ds))
          ic(results_dict)
          # e()
      avg_result_dict["rocauc"] = avg_result_dict["rocauc"] / len(ood_datasets)
      avg_result_dict["aupr_success"] = avg_result_dict["aupr_success"] / len(ood_datasets)
      avg_result_dict["aupr_error"] = avg_result_dict["aupr_error"] / len(ood_datasets)
      avg_result_dict["fpr"] = avg_result_dict["fpr"] / len(ood_datasets)
      
      ic(avg_result_dict)
      result[str(mag)].append(avg_result_dict)
  res_path = os.path.join("./data/results", f"{db}_{cfg.TRAINING.PRT_CONFIG}")
  mkdir(res_path)
  with open(f'{res_path}/avg_{cfg.TRAINING.PRETRAINED}_{cfg.TRAINING.PRT_LAYER}_{cfg.TEST.MAGNITUDE}_temp2.json', 'w') as fp:
    json.dump(avg_result_dict, fp, indent=4)
  with open(f'{res_path}/{cfg.TRAINING.PRETRAINED}_{cfg.TRAINING.PRT_LAYER}_{cfg.TEST.MAGNITUDE}_temp2.json', 'w') as fp:
    json.dump(result, fp, indent=4)

  e()

