import sys
sys.path.append('.')

import argparse
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from imports import * 
from utils import seed_everything, get_args

 

def filter_threshold(llarray, threshold):
  # print(np.sum(llarray < threshold), "outliers")
  return llarray[llarray > threshold]

def plot_hist(loss_vals, ds_lst, thresh, all_labels, fname="test"):
  mpl.rcParams['pdf.fonttype'] = 42
  mpl.rcParams['ps.fonttype'] = 42
  mpl.rcParams['font.family'] = 'Arial'
  fig = plt.figure(figsize=(10,6))
  for i in range(len(loss_vals)):
    sns.histplot(
                filter_threshold(loss_vals[i], thresh), \
                # loss_vals[i], \
                 label=ds_lst[i], \
                  kde=True, stat="density", alpha=.4, edgecolor=(1, 1, 1, .4),\
                # palette=sns.color_palette("viridis", as_cmap=True), 
                # hue=all_labels
                )
    # sns.histplot(filter_threshold(loss_vals[i], -9), label=ds_lst[i], kde=True, stat="density", alpha=.4, edgecolor=(1, 1, 1, .4))
  plt.legend(fontsize=13)
  # plt.xlabel(r'$\log p(x)$', fontsize=24) 
  plt.xlabel(r'$p(x)$', fontsize=24) 
  plt.ylabel('Density', fontsize=24) 
  plt.savefig(f"{fname}.pdf", format="pdf", bbox_inches='tight')
  plt.savefig(f"{fname}.jpg", bbox_inches='tight')




if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  db1 = args.config.split("_")[0]
  
  model = "wideresnet"
  in_dataset = "cifar10"
  n_class = 10 if in_dataset == "cifar10" else 100
  layers = [1, 2, 3, 4] if in_dataset == "cifar10" else [1, 2, 3]
  load_path1 = "./data/baselines/resflow_feat_list/bpds"
  score = 0.00
  score_list = ['0.0', '0.01', '0.005', '0.002', '0.0014', '0.001', '0.0005']
  
  for score in score_list:
    
    for layer in layers:
      # file_name_L3 = os.path.join(f"./data/baselines/resflow_feat_list/{model}_{in_dataset}RealNVP_magnitude", \
      #   'Residual_flow_{}_{}_layer_{}_{}magnitude.npz'.format(in_dataset, in_dataset, layer, score))
      # data_L3 = np.load(file_name_L3)
      # RealNVP_data_L3 = data_L3['arr_0']
      # RealNVP_score_L3 = RealNVP_data_L3[:, 0:n_class]
      
      # bpd1 = np.amax(RealNVP_score_L3, axis=1)
      # ic(bpd1.shape)
      
      bpd1 = np.load(f"{load_path1}/{model}_{in_dataset}_{in_dataset}_test.npy", allow_pickle=True)
      
      # ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
      ood_datasets = ['cifar100']
      
      threshold_dict = {'cifar10_3': -800, 'cifar10_5': -500, 'cifar100_4': -850, 'cifar100_6': -500, 'cifar100_5': -3.9, 'cifar100_6': -1502, 'cifar10_7': -673, 'cifar100_3': -1200}
      threshold_dict_c = {'cifar10_3': -1000, 'cifar10_5': -500, 'cifar100_4': -850, 'cifar100_6': -500, 'cifar100_5': -1400, 'cifar100_6': -1502, 'cifar10_7': -673, 'cifar100_3': -1200}
      
      # all_bpds = [bpd1]
      all_ds = [db1]
      avg_bpds = 0
      # num_ood = bpd1.shape[0]//5
      for ood_ds in ood_datasets:
        
        # file_name_L3 = os.path.join(f"./data/baselines/resflow_feat_list/{model}_{in_dataset}RealNVP_magnitude", \
        #   'Residual_flow_{}_{}_layer_{}_{}magnitude.npz'.format(in_dataset, ood_ds, layer,score))
        # data_L3 = np.load(file_name_L3)
        # RealNVP_data_L3 = data_L3['arr_0']
        # RealNVP_score_L3 = RealNVP_data_L3[:, 0:n_class]
        # bpd2 = np.amax(RealNVP_score_L3, axis=1)
        
        load_path2 = os.path.join(f"{load_path1}", f"{model}_{in_dataset}_{ood_ds}_test.npy")
        bpd2 = np.load(load_path2, allow_pickle=True)
        # bpd2 = bpd2[:2000]
        avg_bpds += bpd2
      
      avg_bpds /= len(ood_datasets)  
      if ood_datasets[0] in ['cifar10', 'cifar100']:
        save_path = os.path.join("./data/plots/resflow", f"{args.config}_{score}_{layer}_{ood_datasets[0]}")
        thresh = threshold_dict_c
      else:
        save_path = os.path.join("./data/plots/resflow", f"{args.config}_{layer}_{score}")
        thresh = threshold_dict
      
      all_bpds = [bpd1, avg_bpds]
      all_labels = ["ID"]* len(bpd1) + ["OOD"] * len(avg_bpds)
      ic(len(all_labels))
      ic(len(bpd1), len(avg_bpds))
      ic(bpd1.min(), bpd1.max())
      ic(avg_bpds.min(), avg_bpds.max())
      # all_ds = [db1, 'avg']
      all_ds = ["ID", 'OOD']
      plot_hist(all_bpds, all_ds, thresh[args.config], all_labels, fname=save_path)
      e()
      
  
