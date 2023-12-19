import torch 
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

from imports import *


class RafDb(Dataset):
  def __init__(self, cfg, mode, train_transform, test_transform) -> None:
    super().__init__()

    # CLASS VARs
    self.cfg = cfg
    self.mode = mode

    # PATHS and DFs
    self.data_dir = os.path.join(cfg.PATHS.DATA_ROOT, "Image", "aligned")
    label_dir = os.path.join(cfg.PATHS.DATA_ROOT, "EmoLabel", "list_patition_label.txt")
    self.labeldf = pd.read_csv(label_dir, sep=" ", header=None)
    self.labeldf.columns = ["fname", "exp"]
    self.labeldf['exp'] = self.labeldf['exp'] - 1

    # PRIVATE DICTS
    self.label_dict = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}
    self.cnt_dict = {"Surprise":0, "Fear":0, "Disgust":0, "Happiness":0, "Sadness":0, "Anger":0, "Neutral": 0}
    self.allowed_labels = [0, 1, 2, 3, 4, 5, 6]
    if len(self.allowed_labels) != cfg.DATASET.N_CLASS:
      raise ValueError("`N_CLASS` is different from `allowed_labels`")
    
    self.all_files_dict = self.getAllFiles()
    self.all_files = list(self.all_files_dict.keys())
    self.all_labels = list(self.all_files_dict.values())

    # AUGs
    self.alb_train, self.alb_val = train_transform, test_transform
    print(self.cnt_dict)

  

  def getAllFiles(self):
    all_files_dict = {}
    for entry1 in os.scandir(self.data_dir):
      fpath = entry1.path
      fname_w_ext = entry1.name
      if os.path.splitext(fname_w_ext)[-1] not in ['.jpg', '.png']:
        continue
      
      fname_w_ext = re.sub("_aligned", "", fname_w_ext)
      fname = os.path.splitext(fname_w_ext)[0]
      fmode = fname.split('_')[0]

      expr = int(self.labeldf.loc[self.labeldf['fname']==fname_w_ext, 'exp'].item())
      label = self.label_dict[expr]
      
      # CONTINUE IF NO CONDITIONS BELOW MET
      if expr not in self.allowed_labels:
        continue
      if self.mode == "train" and fmode != "train":
        continue
      elif self.mode == "val" and fmode != "test":
        continue
      
      # APPEND FILEPATH AND LABEL
      all_files_dict[fpath] = expr
      self.cnt_dict[label] += 1

    return all_files_dict

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    fpath = self.all_files[idx]
    fname = os.path.splitext(fpath)[0].split('/')[-1]
    label = self.all_files_dict[fpath]
    image = Image.open(self.all_files[idx])
    if self.mode== "train":
      transform = self.alb_train
    else:
      transform = self.alb_val
    image_aug = transform(image)

    return image_aug, label
