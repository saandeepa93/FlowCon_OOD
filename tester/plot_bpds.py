import sys
sys.path.append('.')

import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from imports import * 
from utils import seed_everything, get_args

 

def filter_threshold(llarray, threshold):
  # print(np.sum(llarray < threshold), "outliers")
  return llarray[llarray > threshold]

def plot_hist(loss_vals, ds_lst, thresh, fname="test"):
  for i in range(len(loss_vals)):
    # sns.histplot(filter_threshold(loss_vals[i], thresh), label=ds_lst[i], kde=True, stat="density", alpha=.4, edgecolor=(1, 1, 1, .4))
    sns.histplot(filter_threshold(loss_vals[i], -9), label=ds_lst[i], kde=True, stat="density", alpha=.4, edgecolor=(1, 1, 1, .4))
  plt.legend(fontsize=13)
  plt.savefig(fname, bbox_inches='tight')




if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  db1 = args.config.split("_")[0]
  
  load_path1 = os.path.join("./data/bpds", args.config)
  bpd1 = np.load(f"{load_path1}/{args.config}_test.npy", allow_pickle=True)
  
  ood_datasets = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
  # ood_datasets = ['aff']
  
  threshold_dict = {'raf_4': -5, 'raf_5': -15, 'aff_4': -5, 'aff_5': -18, 'cifar10_3': -4, 'cifar10_4': -10}
  
  all_bpds = [bpd1]
  all_ds = [db1]
  avg_bpds = 0
  num_ood = bpd1.shape[0]//5
  for ood_ds in ood_datasets:
    load_path2 = os.path.join(f"./data/bpds/{args.config}", f"{ood_ds}_test.npy")
    bpd2 = np.load(load_path2, allow_pickle=True)
    bpd2 = bpd2[:num_ood]
    avg_bpds += bpd2
    
    # all_bpds.append(bpd2)
    # all_ds.append(ood_ds)
  
  avg_bpds /= len(ood_datasets)  
  if ood_datasets[0] in ['raf', 'aff']:
    save_path = os.path.join("./data/plots", f"{args.config}_{ood_datasets[0]}.png")
  else:
    save_path = os.path.join("./data/plots", f"{args.config}_600.png")
  
  all_bpds = [bpd1, avg_bpds]
  # all_ds = [db1, 'avg']
  all_ds = ["ID", 'OOD']
  plot_hist(all_bpds, all_ds, threshold_dict[args.config], fname=save_path)
    
  
