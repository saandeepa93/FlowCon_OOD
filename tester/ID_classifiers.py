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

import itertools
from einops import rearrange
from math import log, pi

from loaders import select_ood_testset, select_classifier, select_dataset, select_ood_transform, select_transform
from utils import seed_everything, get_args, get_metrics, mkdir, make_roc, plot_umap, metric
from configs import get_cfg_defaults
from models import *
from losses import FlowConLoss
from sklearn.metrics import accuracy_score

import torch.profiler as profiler


def get_metrics_ood(label, score, invert_score=False):
    results_dict = {}
    if invert_score:
        score = score - score.max()
        score = np.abs(score)

    error = 1 - label
    fpr = 0
    eval_range = np.arange(score.min(), score.max(), (score.max() - score.min()) / 10000)
    for i, delta in enumerate(eval_range):
        tpr = len(score[(label == 1) & (score >= delta)]) / len(score[(label == 1)])
        if 0.9505 >= tpr >= 0.9495:
            print(delta)
            break
    print(delta)
    e()


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


def calc_scores_2(cfg, args, loader, pretrained, flow, cls, mu, log_sd, criterion, loss_criterion, device, flg=False):
  n_pixel = cfg.FLOW.IN_FEAT
  scores_all = []
  labels_all = []
  preds_all = []
  fnames_all = []

  pretrained.eval()
  flow.eval()
  cls.eval()
  for b, (x, true_label) in enumerate(tqdm(loader), 0):
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
    loss = loss_criterion(log_probs, label)
    
    # loss = torch.mean(-torch.max(log_probs, dim=-1)[0])
    # loss = loss_criterion(F.softmax(log_probs, dim=-1), label)

    loss.backward()

    # INPUT PREPROCESSING
    # NEG. FGSM METHOD
    gradient =  torch.ge(x.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    with torch.no_grad():
      if cfg.TRAINING.PRETRAINED == 'wideresnet':
          gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.5))
      elif cfg.TRAINING.PRETRAINED == 'resnet18':
          gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.5))
          gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.5))
      noisy_x = torch.add(x.data, -cfg.TEST.MAGNITUDE, gradient)
      
      # _, _, features = pretrained.penultimate_forward(noisy_x)
      features = pretrained.intermediate_forward(noisy_x, cfg.TRAINING.PRT_LAYER)
      features = features.view(features.size(0), features.size(1), -1)
      features = torch.mean(features, 2)

      z, n_mu, n_log_sd, sdlj, _, _ = flow(features)
      score, _, _, _ = criterion.nllLoss(z, sdlj, n_mu, n_log_sd) #CIFAR10_3_RESNET

      score = score.detach().cpu().tolist()
      true_label = true_label.detach().cpu().tolist()
      label = label.detach().cpu().tolist()

      scores_all += score
      preds_all += label
      labels_all += true_label


  return labels_all, preds_all, scores_all



if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")
  ckp_path = os.path.join(f"./checkpoints/{db}")
  # ckp_path = os.path.join(f"./checkpoints/main")
  # ckp_path = os.path.join(f"./checkpoints/{db}/{args.config}")
  mkdir(ckp_path)

  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.TRAINING.BATCH=192
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

  # id_transform = select_ood_transform(cfg, cfg.DATASET.IN_DIST, cfg.DATASET.IN_DIST)
  # test_set = select_ood_testset(cfg, cfg.DATASET.IN_DIST, id_transform)
  id_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=False)
  test_set = select_dataset(cfg, cfg.DATASET.IN_DIST, id_transform, train=False)
  orig_num_ood = len(test_set) // 5
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)


  # ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
  ood_datasets = ['aff']
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
      # flow_ckp = torch.load(f"{ckp_path}/{args.config}/{db}_{cfg.TRAINING.PRT_CONFIG}_{cfg.TRAINING.PRETRAINED}_layer{cfg.TRAINING.PRT_LAYER}_flow_200.pt", map_location=device)
      # flow_ckp = torch.load(f"{ckp_path}/{db}_{cfg.TRAINING.PRT_CONFIG}_{cfg.TRAINING.PRETRAINED}_layer{cfg.TRAINING.PRT_LAYER}_flow_600.pt", map_location=device)
      sd = {k: v for k, v in flow_ckp['state_dict'].items()}
      state = model.state_dict()
      state.update(sd)
      model.load_state_dict(state, strict=True)
      print("Total Trainable Parameters of flow model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

      dist_dir = f"./data/distributions/{args.config}"
      mkdir(dist_dir)
      dist_path = os.path.join(dist_dir, f"layer{cfg.TRAINING.PRT_LAYER}")
      mkdir(dist_path)

      
      # load params
      mu = torch.load(os.path.join(dist_path, "mu.pt"), map_location=device)
      log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"), map_location=device)
      mu =  torch.stack(mu)
      log_sd =  torch.stack(log_sd)

      thresh_dict = {"aff_4": -2.796689883709161, "aff_5": -11.944983131679866, "raf_4": -2.917621004939253, "raf_5": -8.999182610697972}
      
      for ood_ds in ood_datasets:
        if cfg.DATASET.IN_DIST == ood_ds:
          continue
        print(f"Running {ood_ds}")
        if cfg.TEST.SCORE:
          ood_transform = select_ood_transform(cfg, ood_ds, cfg.DATASET.IN_DIST)
          ood_dataset = select_ood_testset(cfg, ood_ds, ood_transform)
          ood_loader = DataLoader(ood_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

          num_ood = min(orig_num_ood, len(ood_dataset))
          
          print("Calculating Scores...")
          labels_all, preds_all, scores_all = calc_scores_2(cfg, args, ood_loader, pretrained, model, cls, mu, log_sd, criterion, loss_criterion, device, False)
          ood_labels = np.zeros(len(scores_all))
          scores_all = np.array(scores_all)

          thresh = scores_all.mean()#thresh_dict[args.config]
          id_ind = [i for i, s in enumerate(scores_all) if s < thresh]
          # id_ind = [i for i, s in enumerate(scores_all) if s > thresh]
          # id_ind = [i for i, s in enumerate(scores_all)]
          gt = [labels_all[i] for i in id_ind]
          pr = [preds_all[i] for i in id_ind]

          if ood_ds == "raf":
            ad_prs = []
            pr_dict = {0:6, 2:4, 3:0, 4:1, 5:2, 6:5}
            for p in pr:
              if p in [1, 7]:
                ad_prs.append(3)
              else:
                ad_prs.append(pr_dict[p])
            print(len(gt), len(ad_prs))
            print(accuracy_score(gt, ad_prs))
          elif ood_ds == "aff":
            ad_prs = []
            ad_gt = [i if i <=6 else 1 for i in gt ]

            pr_dict = {0:3, 1: 4, 2:5, 3:1, 4:2, 5:6, 6:0}
            for p in pr:
              ad_prs.append(pr_dict[p])

            print(len(gt), len(ad_prs))
            print(accuracy_score(ad_gt, ad_prs))
          e()




