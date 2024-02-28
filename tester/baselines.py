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

from loaders import select_ood_testset, select_classifier, select_dataset, select_ood_transform, select_transform, select_ood_baseline_transform, select_baseline_transform
from utils import seed_everything, get_args, get_metrics, mkdir, make_roc, plot_umap, metric, get_metrics_ood
from configs import get_cfg_defaults
from models import *
from score import get_score
import lib_generation
from sklearn.linear_model import LogisticRegressionCV


def get_Xy(trainloader):
  data_list = []
  target_list = []

  # Loop over the DataLoader
  for inputs, targets in trainloader:
      # Move the batch to CPU if necessary and convert to numpy
      data_list.append(inputs.cpu().numpy())
      target_list.append(targets.cpu().numpy())

  # Concatenate all the batches together
  X_train = np.concatenate(data_list, axis=0)
  y_train = np.concatenate(target_list, axis=0)
  return X_train, y_train


def compute_traditional_ood(base_dir, in_dataset, out_datasets, method, name, args=None):
    avg_result_dict = {"rocauc": 0, "aupr_success": 0, "aupr_error": 0, "fpr": 0}
    known = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name))
    result = {}
    for out_dataset in out_datasets:
      novel = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset))
      # print(known.shape, novel.shape, out_dataset, in_dataset)
      novel = novel[:len(known)//5]
      scores = np.concatenate([np.array(known), np.array(novel)], axis=0)
      labels = np.concatenate([np.ones(len(known)), np.zeros(len(novel))], axis=0)
      # if method == "mahalanobis":
      #    lr = LogisticRegressionCV(n_jobs=-1).fit(scores, labels)
      results_dict = get_metrics_ood(labels, scores, invert_score=False)
      avg_result_dict["rocauc"] += results_dict["rocauc"]
      avg_result_dict["aupr_success"] += results_dict["aupr_success"]
      avg_result_dict["aupr_error"] += results_dict["aupr_error"]
      avg_result_dict["fpr"] += results_dict["fpr"]

      # result[str(mag)].append(metric(np.array(scores_in), np.array(scores_out), stype=ood_ds))
      ic(results_dict)
      result[out_dataset] = results_dict
          # e()
    avg_result_dict["rocauc"] = avg_result_dict["rocauc"] / len(out_datasets)
    avg_result_dict["aupr_success"] = avg_result_dict["aupr_success"] / len(out_datasets)
    avg_result_dict["aupr_error"] = avg_result_dict["aupr_error"] / len(out_datasets)
    avg_result_dict["fpr"] = avg_result_dict["fpr"] / len(out_datasets)
    ic(avg_result_dict)
    res_path = os.path.join(f"./data/results/baseline/{method}", f"{args.config}")
    mkdir(res_path)
    with open(f'{res_path}/avg_{cfg.TRAINING.PRETRAINED}_{cfg.TRAINING.PRT_LAYER}.json', 'w') as fp:
      json.dump(avg_result_dict, fp, indent=4)
    with open(f'{res_path}/{cfg.TRAINING.PRETRAINED}_{cfg.TRAINING.PRT_LAYER}.json', 'w') as fp:
      json.dump(result, fp, indent=4)


def react(cfg, in_loader, out_datasets, pretrain, cls, in_dataset, method, method_args, forward_func=None ):
  in_save_dir = f"./data/baselines/{in_dataset}/{method}/{cfg.TRAINING.PRETRAINED}"
  mkdir(in_save_dir)
  N = len(in_loader.dataset)
  num_ood = N // 5
  count = 0
  f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
  g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

  for j, data in enumerate(in_loader):
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    curr_batch_size = images.shape[0]

    inputs = images.float()

    with torch.no_grad():
      # logits = cls(pretrain(inputs))
      logits = forward_func(inputs, pretrain, cls, method)
      outputs = F.softmax(logits, dim=1)
      outputs = outputs.detach().cpu().numpy()
      preds = np.argmax(outputs, axis=1)
      confs = np.max(outputs, axis=1)

      for k in range(preds.shape[0]):
          g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))
    scores = get_score(cfg, inputs, pretrain, cls, forward_func, method, method_args, logits=logits)
    for score in scores:
      f1.write("{}\n".format(score))

    count += curr_batch_size
    # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
    t0 = time.time()

  f1.close()
  g1.close()

   # OOD evaluation
  for out_dataset in out_datasets:

      out_save_dir = os.path.join(in_save_dir, out_dataset)
      mkdir(out_save_dir)

      f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

  ###################################Out-of-Distributions#####################################
      t0 = time.time()
      print("Processing out-of-distribution images")
      ood_transform = select_ood_transform(cfg, out_dataset, cfg.DATASET.IN_DIST)
      ic(ood_transform)
      ood_dataset = select_ood_testset(cfg, out_dataset, ood_transform)
      ood_loader = DataLoader(ood_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)
      N = len(ood_loader.dataset)
      count = 0
      for j, data in enumerate(ood_loader):

          images, labels = data
          images = images.cuda()
          curr_batch_size = images.shape[0]

          inputs = images.float()

          with torch.no_grad():
            # logits = cls(pretrain(inputs))
            logits = forward_func(inputs, pretrain, cls, method)

          ood_scores = get_score(cfg, inputs, pretrain, cls, forward_func, method, method_args, logits=logits)
          # scores = scores[:num_ood]
          for score in ood_scores:
              f2.write("{}\n".format(score))

          count += curr_batch_size
          print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
          t0 = time.time()

      f2.close()



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
  cfg.TRAINING.BATCH=192
  print(cfg)
  
  methods = ["react", "energy", "mahalanobis", "msp", "odin" ]
  # methods = ["mahalanobis"]
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
  pretrained.eval(), cls.eval()


  # id_transform = select_ood_transform(cfg, cfg.DATASET.IN_DIST, cfg.DATASET.IN_DIST)
  # test_set = select_ood_testset(cfg, cfg.DATASET.IN_DIST, id_transform)
  id_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=False)
  ic(id_transform)
  test_set = select_dataset(cfg, cfg.DATASET.IN_DIST, id_transform, train=False)
  num_ood = len(test_set) // 5
  train_set = select_dataset(cfg, cfg.DATASET.IN_DIST, id_transform, train=True)

  train_loader = DataLoader(train_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

  for method in methods:
    threshold = 1e6
    print(f"METHOD: {method}")
    # FOR MAHALANOBIS
    method_args = {
      "num_classes": cfg.DATASET.N_CLASS,
      "magnitude": .0024, 
      'temperature': 1.0,
      "regressor": None
      }

    forward_func = None
    if method == "mahalanobis":
      temp_x = torch.rand(2,3,32,32).cuda()
      temp_x = Variable(temp_x)
      temp_list = pretrained.feature_list(temp_x)
      num_output = len(temp_list)
      feature_list = np.empty(num_output)
      count = 0
      for out in temp_list:
          feature_list[count] = out.size(1)
          count += 1
      sample_mean, precision = lib_generation.sample_estimator(pretrained, cls, cfg.DATASET.N_CLASS, feature_list, train_loader)
      method_args = {
        "sample_mean": sample_mean,
        "precision": precision,
        "num_output": len(feature_list),
        "num_classes": cfg.DATASET.N_CLASS,
        "magnitude": 0, 
        "regressor": None
      }

    # FOR MAHALANOBIS

    if method == "odin":
      method_args = {
      "num_classes": cfg.DATASET.N_CLASS,
      "magnitude": .04, 
      'temperature': 1.0,
      "regressor": None
      }

    
    elif method == "react":
      threshold=1.0

      # M_train_scores = []
      # for j, data in enumerate(tqdm(train_loader)):
      #   images, labels = data
      #   images = images.cuda()
      #   curr_batch_size = images.shape[0]
      #   inputs = images.float()

      #   with torch.no_grad():
      #     logits = pretrained(inputs)
      #   M_train_scores.append(logits.cpu().numpy())
      # M_train_scores = np.concatenate(M_train_scores, axis=0)
      # ic(M_train_scores.shape)
      # print(f"\nTHRESHOLD at percentile {90} is:")
      # print(np.percentile(M_train_scores.flatten(), 90))
      # e()
    

    def forward_fun():
      def forward_threshold(inputs, model, cls, method):
          h_x = model(inputs)
          if method == "react":
            h_x = h_x.clip(max=1.6)
          else:
            h_x = h_x.clip(max=1e6)
          logits = cls(h_x)
          return logits
      return forward_threshold
    
    forward_func = forward_fun()
    # ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
    ood_datasets = ['cifar100']
    react(cfg, test_loader, ood_datasets, pretrained, cls, cfg.DATASET.IN_DIST,  method, method_args, forward_func)
    compute_traditional_ood("./data/baselines", cfg.DATASET.IN_DIST, ood_datasets, method, cfg.TRAINING.PRETRAINED, args=args)


    