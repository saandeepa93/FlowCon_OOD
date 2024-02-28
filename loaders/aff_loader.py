import torch 
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import pandas as pd
import numpy as np
import glob
from collections import Counter
from sys import exit as e
from icecream import ic


class AffectDataset(Dataset):
  def __init__(self, cfg, mode,  train_transform, test_transform):
    super().__init__()

    self.cfg = cfg
    self.mode = mode
    
    root_dir_a = "/dataset/AffectNet"
    if mode == "train":
      root_dir = os.path.join(root_dir_a, "train_set")
    elif mode == "val":
      root_dir = os.path.join(root_dir_a, "val_set")
    self.root_dir = root_dir

    # Private dicts
    self.label_dict = {"Neutral": 0, "Happiness":1, "Sadness":2, "Surprise":3, "Fear":4, "Disgust":5, "Anger":6, "Contempt": 7}
    self.label_dict_inverse = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
    self.cnt_dict = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}                       
    self.allowed_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    # self.allowed_labels = [7]
    self.all_files_dict = self.__getAllFiles__()

    self.all_files = list(self.all_files_dict.keys())
    self.alb_train, self.alb_val = train_transform, test_transform
    self.all_labels = list(self.all_files_dict.values())
    ic(Counter(self.all_labels))
  

  def __getAllFiles__(self):
    all_files = {}
    nav_dir = os.path.join(self.root_dir, "images")
    for entry in os.scandir(nav_dir):
      fname_w = entry.name
      fpath = entry.path
      if os.path.splitext(fname_w)[-1] != ".jpg":
        continue
      fname = fname_w.split(".")[0]
      exp = int(np.load(os.path.join(self.root_dir, "annotations", fname+"_exp.npy")).item())
      if exp not in self.allowed_labels:
        continue
        
      all_files[fpath] = exp
      self.cnt_dict[exp] += 1
    return all_files

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    fpath = self.all_files[idx]
    fname = fpath.split("/")[-1].split(".")[0]

    # image = np.array(Image.open(self.all_files[idx]))
    image = Image.open(self.all_files[idx])
    exp = self.all_files_dict[fpath]
    
    if self.mode== "train":
        transform = self.alb_train
    else:
      transform = self.alb_val
    
    image_aug = transform(image)
    
    return image_aug, exp