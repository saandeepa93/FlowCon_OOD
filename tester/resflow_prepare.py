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
from torch.autograd import Variable

from loaders import select_ood_testset, select_classifier, select_dataset, select_ood_transform, select_transform
from utils import *
from configs import get_cfg_defaults
from models import *
from losses import FlowConLoss


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
  # cfg.freeze()
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

  # for param in pretrained.parameters():
  #   param.requires_grad=False
  pretrained.eval()
  cls.eval()

  train_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=True)
  test_transform = select_transform(cfg, cfg.DATASET.IN_DIST, pretrain=False)
  
  train_set = select_dataset(cfg, cfg.DATASET.IN_DIST, test_transform, train=True)
  train_loader = DataLoader(train_set, batch_size=cfg.TRAINING.BATCH, shuffle=True, num_workers = cfg.DATASET.NUM_WORKERS)

  test_set = select_dataset(cfg, cfg.DATASET.IN_DIST, test_transform, train=False)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

  with torch.no_grad():

    # set information about feature extraction
    temp_x = torch.rand(2, 3, 32, 32).cuda()
    temp_x = Variable(temp_x)
    temp_list = pretrained.feature_list(temp_x)
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(cfg.DATASET.N_CLASS)
    num_sample_per_class.fill(0)
    list_features = []
    list_features_test = []
    list_features_out = []
    for i in range(num_output):
        temp_list = []
        list_features_test.append(0)
        list_features_out.append(0)
        for j in range(cfg.DATASET.N_CLASS):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        out_features = pretrained.feature_list(data)

        features = out_features[-1]
        features = features.view(features.size(0), features.size(1), -1)
        features = torch.mean(features, 2)
        output = cls(features)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(cfg.DATASET.N_CLASS, int(num_feature)).cuda()
        for j in range(cfg.DATASET.N_CLASS):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    A = []
    A_inv = []
    log_abs_det_A_inv = []
    for k in range(num_output):
        X = 0
        for i in range(cfg.DATASET.N_CLASS):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        u, s, vh = np.linalg.svd((X.cpu().numpy())/ np.sqrt(X.shape[0]), full_matrices=False)
        covariance_real = np.cov(X.cpu().numpy().T)
        valid_indx = s > 1e-5
        if (valid_indx.sum() % 2 > 0):
            valid_indx[valid_indx.sum()-1] = False
        covriance_cal = np.matmul(np.matmul(vh[valid_indx, :].transpose(), np.diag(s[valid_indx] ** 2)), vh[valid_indx, :])
        A_temp = np.matmul(vh[valid_indx, :].transpose(), np.diag(s[valid_indx]))
        A.append(A_temp)
        covriance_cal2 = np.matmul(A_temp, A_temp.transpose())
        s_inv = 1/s[valid_indx]
        A_inv_temp = np.matmul(np.diag(s_inv), vh[valid_indx, :])
        A_inv.append(A_inv_temp)
        log_abs_det_A_inv_temp = np.sum(np.log(np.abs(s_inv)))
        log_abs_det_A_inv.append(log_abs_det_A_inv_temp)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100.0 * int(correct) / int(total)))

    num_sample_per_output = np.empty(num_output)
    num_sample_per_output.fill(0)
    for data, target in test_loader:

        data = data.cuda()
        data = Variable(data, volatile=True)
        out_features = pretrained.feature_list(data)
        features = out_features[-1]
        features = features.view(features.size(0), features.size(1), -1)
        features = torch.mean(features, 2)
        output = cls(features)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

            if  num_sample_per_output[i] ==0:
                list_features_test[i] = out_features[i]
            else:
                list_features_test[i] = torch.cat((list_features_test[i], out_features[i]), 0)
            num_sample_per_output[i] += 1



    # out_dist_list = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
    out_dist_list = ['imagenet_resize']
    for out_dist in out_dist_list:
        ood_transform = select_ood_transform(cfg, out_dist, cfg.DATASET.IN_DIST)
        ood_dataset = select_ood_testset(cfg, out_dist, ood_transform)
        out_test_loader = DataLoader(ood_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)

        num_sample_per_output.fill(0)

        for data, target in out_test_loader:

            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=True), Variable(target)
            out_features = pretrained.feature_list(data)
            features = out_features[-1]
            features = features.view(features.size(0), features.size(1), -1)
            features = torch.mean(features, 2)
            output = cls(features)
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

                if num_sample_per_output[i] == 0:
                    list_features_out[i] = out_features[i]
                else:
                    list_features_out[i] = torch.cat((list_features_out[i], out_features[i]), 0)
                num_sample_per_output[i] += 1

        for i in range(num_output):
            sample_class_mean[i] = sample_class_mean[i].cpu()
            list_features_test[i] = list_features_test[i].cpu()
            list_features_out[i] = list_features_out[i].cpu()
            for j in range(cfg.DATASET.N_CLASS):
                list_features[i][j] = list_features[i][j].cpu()

        file_name = os.path.join('./data/baselines/resflow_feat_list', 'feature_lists_{}_{}_{}_Wlinear.pickle'.format(cfg.TRAINING.PRETRAINED, out_dist, cfg.DATASET.IN_DIST))
        with open(file_name, 'wb') as f:
            pickle.dump([sample_class_mean, list_features, list_features_test, list_features_out, A, A_inv, log_abs_det_A_inv] , f)
