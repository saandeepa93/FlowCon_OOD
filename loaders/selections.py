import os
from torchvision import datasets, transforms
from PIL import Image
import torch
from torch import nn 
import timm

from models import ResNet18, ResNet50, ResNet_Linear
from models import Resnet101, Resnet101_Linear, EfficientNet
from models import Wide_ResNet, WideResNet_Linear
from .raf_loader import RafDb
from .aff_loader import AffectDataset





def select_baseline_transform(cfg, dataset, pretrain=False):
    if pretrain:
        if dataset in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset == 'tinyimagenet':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['raf', 'aff']:
        
            if cfg.TRAINING.PRETRAINED == 'resnet101':
                transform = transforms.Compose([
                    transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation([-45, 45]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            else:    
                transform = transforms.Compose([
                transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([-45, 45]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
            ])
        else:
            raise NotImplementedError
    else:
        if dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['raf', 'aff']:
           transform = transforms.Compose([
            transforms.Resize(cfg.DATASET.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise NotImplementedError
      

    return transform





def select_transform(cfg, dataset, pretrain=False):
    if pretrain:
        if dataset in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset == 'tinyimagenet':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['raf', 'aff']:
        
            if cfg.TRAINING.PRETRAINED == 'resnet101':
                transform = transforms.Compose([
                    transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation([-45, 45]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            else:    
                transform = transforms.Compose([
                transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([-45, 45]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
            ])
        else:
            raise NotImplementedError
    else:
        if dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['raf', 'aff']:
            if cfg.TRAINING.PRETRAINED == 'resnet101':
                    transform = transforms.Compose([
                        transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            else:    
                transform = transforms.Compose([
                transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]) 
                ])
        else:
            raise NotImplementedError
      

    return transform


def select_dataset(cfg, dataset, transform, train):
    if dataset == 'cifar10':
        trainset = datasets.CIFAR10(os.getcwd() + './data/', train=train, transform=transform, download=True)
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100(os.getcwd() + './data/', train=train, transform=transform, download=True)
    elif dataset == 'tinyimagenet':
        if train:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/tiny-imagenet-200/train/',
                                            transform=transform)
        else:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/tiny-imagenet-200/test/',
                                            transform=transform)
    elif dataset == 'tinyimages':
        if train:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/TinyImages-100000/', transform=transform)
        else:
            raise RuntimeError('Only available as ood training set')
    elif dataset == 'places365':
        if train:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/Places365/', transform=transform)
        else:
            raise RuntimeError('Only available as ood training set')
    elif dataset == 'raf':
      mode = "train" if train else "val"
      trainset = RafDb(cfg, mode, transform, transform)
    elif dataset == 'aff':
      mode = "train" if train else "val"
      trainset = AffectDataset(cfg, mode, transform, transform)
    else:
        raise NotImplementedError
    return trainset


def select_classifier(cfg, dataset, arch, num_classes, fw_layers=1, depth=40, width=2):
    if dataset == 'cifar10':
        if arch == 'resnet18':
            enc = ResNet18(fw_layers=fw_layers)
            cls = ResNet_Linear()
        elif arch == 'wideresnet':
            enc = Wide_ResNet(depth, width, 0.3, fw_layers=fw_layers)
            cls = WideResNet_Linear(width, num_classes=num_classes)
        else:
            raise NotImplementedError
    elif dataset == 'cifar100':
        if arch == 'resnet18':
            enc = ResNet18(fw_layers=fw_layers)
            cls = ResNet_Linear(num_classes=num_classes)
        elif arch == 'wideresnet':
            enc = Wide_ResNet(depth, width, 0.3, fw_layers=fw_layers)
            cls = WideResNet_Linear(width, num_classes=num_classes)
        else:
            raise NotImplementedError
    elif dataset in ['raf', 'aff']:
        if arch == 'resnet18':
            enc = ResNet18(fw_layers=fw_layers)
            cls = ResNet_Linear(num_classes=num_classes)
        elif arch == 'wideresnet':
            enc = Wide_ResNet(depth, width, 0.3, fw_layers=fw_layers)
            cls = WideResNet_Linear(width, num_classes=num_classes)
        elif arch == 'resnet101':
            enc = Resnet101()
            cls = Resnet101_Linear(cfg)
        elif arch == 'effnet':
            enc = EfficientNet()
            cls = Resnet101_Linear(cfg)
        else:
            raise NotImplementedError
    elif dataset == 'tinyimagenet':
        if arch == 'resnet50':
            enc = ResNet50(fw_layers=fw_layers, img_size=64)
            cls = ResNet_Linear(num_classes=num_classes, expansion=4)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return enc, cls



def select_ood_testset(cfg, dataset, transform):
    if dataset == 'textures':
        print(transform)
        ood_data = datasets.ImageFolder('/dataset/dtd/images/', transform=transform)
    elif dataset == 'svhn':
        ood_data = datasets.SVHN(os.getcwd() + './data/', split='test', transform=transform, download=True)
    elif dataset == 'places365':
        ood_data = datasets.ImageFolder('/dataset/Places/', transform=transform)
    elif dataset == 'lsun-c':
        ood_data = datasets.ImageFolder('/dataset/LSUN/', transform=transform)
    elif dataset == 'lsun-r':
        ood_data = datasets.ImageFolder('/dataset/LSUN_resize/', transform=transform)
    elif dataset == 'isun':
        ood_data = datasets.ImageFolder('/dataset/iSUN/', transform=transform)
    elif dataset == 'cifar10':
        ood_data = datasets.CIFAR10(os.getcwd() + './data/', transform=transform, train=False)
    elif dataset == 'cifar100':
        ood_data = datasets.CIFAR100(os.getcwd() + './dataset/', transform=transform, train=False, download=True)
    elif dataset == 'tinyimagenet':
        ood_data = datasets.ImageFolder('/dataset/tiny-imagenet-200/test/', transform=transform)
    elif dataset == 'lsun':
        ood_data = datasets.ImageFolder('/dataset/SUN/', transform=transform)
    elif dataset == 'inaturalist':
        ood_data = datasets.ImageFolder('/dataset/iNaturalist/', transform=transform)
    elif dataset == 'raf':
        ood_data = RafDb(cfg, "val", transform, transform)
    elif dataset == 'aff':
        ood_data = AffectDataset(cfg, "val", transform, transform)
    elif dataset == "imagenet_resize":
        ood_data = datasets.ImageFolder('/dataset/Imagenet_resize/', transform=transform)
    else:
        raise NotImplementedError
    return ood_data



def select_ood_baseline_transform(cfg, dataset, id_dataset='cifar10'):
    if id_dataset in ['cifar10', 'cifar100', 'raf', 'aff']:
        if dataset in ['textures', 'places365', 'lsun-c']:
            transform = transforms.Compose([
                # transforms.Resize(32),
                transforms.Resize(cfg.DATASET.IMG_SIZE),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['svhn', 'lsun-r', 'isun', 'cifar10', 'cifar100', 'tinyimages', 'aff', 'raf']:
            transform = transforms.Compose([
                transforms.Resize(cfg.DATASET.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise NotImplementedError

    elif id_dataset == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise NotImplementedError
    return transform



def select_ood_transform(cfg, dataset, id_dataset='cifar10'):
    if id_dataset in ['cifar10', 'cifar100']:
        if dataset in ['textures', 'places365', 'lsun-c']:
            transform = transforms.Compose([
                # transforms.Resize(32),
                transforms.Resize(cfg.DATASET.IMG_SIZE),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['svhn', 'lsun-r', 'isun', 'cifar10', 'cifar100', 'tinyimages', 'imagenet_resize']:
            transform = transforms.Compose([
                transforms.Resize(cfg.DATASET.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['aff']:
          transform = transforms.Compose([
                transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) 
            ])
        else:
            raise NotImplementedError
    elif id_dataset in ['raf', 'aff']:
        if cfg.TRAINING.PRETRAINED == 'resnet101':
                transform = transforms.Compose([
                    transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            if dataset  in ['textures', 'places365', 'lsun-c']:
                transform = transforms.Compose([
                    transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                    transforms.CenterCrop(cfg.DATASET.IMG_SIZE),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
                ])
            elif dataset in ['svhn', 'lsun-r', 'isun', 'cifar10', 'cifar100', 'tinyimages', 'aff', 'raf', 'imagenet_resize']:
                transform = transforms.Compose([
                    transforms.Resize(cfg.DATASET.IMG_SIZE, interpolation=Image.BILINEAR),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
                ])

    elif id_dataset == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise NotImplementedError
    return transform
