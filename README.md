<p align="center">

  <h2 align="center"><strong>FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning</strong></h2>

  <h3 align="center"><span style="font-size:1em;" color><strong>ECCV 2024</strong></span>
  </h3>

  <p align="center">
    <a href="https://saandeepa93.github.io/"><strong> Saandeep Aathreya</strong></a>,
    <a href="https://scanavan.github.io/"><strong> Shaun Canavan</strong></a>
    <br>
    <span style="font-size:1em; "><strong> University of South Florida, USA</strong>.</span>
    <br>
  </p>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2407.03489" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-Paper-greem.svg">
  </a> 
</p>


## :bulb: **Contributions**:

<p align="center">
  <img src="docs/static//images/contrib.png" width="410" />
</p>

<!-- ![main-method](docs/static//images/contrib.png) -->

- A new density-based OOD detection technique called FlowCon is proposed. We introduce a new loss function $L_{con}$ which contrastively learns class separability in the probability distribution space. This learning occurs without any external OOD dataset and it operates on fixed classifiers.

- The proposed method is evaluated on various metrics - FPR95, AUROC, AUPR-Success, and AUPR-Error and compared against state of the art. We observe that FlowCon is competitive or outperforms most methods under different OOD conditions. Additionally, FlowCon is stable even for a large number of classes and shows improvement for high-dimensional features

- Histogram plots are detailed along with unified manifold approximations (UMAP) embeddings of the trained FlowCon model to respectively showcase it’s OOD detection and class-preserving capabilities. We also show FlowCon’s discriminative capabilities.



## **Results**
<!-- 

### **FAR-OOD likelihood plots when $D_{in}=CIFAR10$ on ResNet-18 and WideResNet models**
<p align="center">
  <img src="docs/static/images/ll_cifar10.png" width="410" height="400" />
</p>

### **FAR-OOD likelihood plots when $D_{in}=CIFAR100$ on ResNet-18 and WideResNet models**
<p align="center">
  <img src="docs/static/images/ll_cifar100.png" width="410" height="400" />
</p>

## **FlowCon Comparison for Semantic and Covariate Shift**
 -->

</br>

<p align="center">
  <img src="docs/static/images/main_res.png" width="400" height="350"/>
</p>

<div style="display: flex; justify-content: space-between;">
  <img src="docs/static/images/far_near.png" alt="Image 1" style="width: 45%;"/>
  <img src="docs/static/images/near.png" alt="Image 2" style="width: 45%; "/>
</div>
<!-- <p align="center">
  <img src="docs/static/images/far_near.png" width="310" height="200"/>
</p>

<!-- <p align="center">
  <img src="docs/static/images/near.png" width="310" height="200"/>
</p> -->

<!-- ## **UMAP Embeddings for Semantic and Covariate Shift**
<p align="center">
  <img src="docs/static/images/umap.png" width="410" height="200"/>
</p>

<p align="center">
  <img src="docs/static/images/umap_cifar100.png" width="410" height="200"/>
</p> -->




## **Configurations**
* All config files are present under path `./configs/experiments`
* We have 4 config files 
```
cifar10_3.yaml -> CIFAR10 with ResNet18
cifar10_5.yaml -> CIFAR10 with WideResNet
cifar100_6.yaml -> CIFAR100 with ResNet18
cifar100_4.yaml -> CIFAR100 with WideResNet
```

## **Pretrained Classifiers**

* The original ResNet18 and WideResNet classifiers are taken from [Heatmap paper](https://github.com/jhornauer/heatmap_ood?tab=readme-ov-file#pre-trained-models)



## **Training FlowCon model**

### **CIFAR-10**
Training `Resnet` and `WideResNet` FlowCon model for CIFAR10. Follow similar commands for other configs

```
CUDA_VISIBLE_DEVICES=0 python ./trainer/train_flow.py --config cifar10_3 #Resnet

CUDA_VISIBLE_DEVICES=0 python ./trainer/train_flow.py --config cifar10_5 #WRN
```

### **CIFAR-100**
```
CUDA_VISIBLE_DEVICES=0 python ./trainer/train_flow.py --config cifar100_6
```

```
CUDA_VISIBLE_DEVICES=0 python ./trainer/train_flow.py --config cifar100_4
```


## **Getting OOD Scores**
* Inference follows similar arguments
```
CUDA_VISIBLE_DEVICES=0 python ./tester/emperical_params.py --config cifar10_5

CUDA_VISIBLE_DEVICES=0 python ./tester/emperical_params.py --config cifar100_4
```

* The $k$ distributions will be saved under `./data/distributions` directory

* The results will be saved under `./data/results` directory

## **Pre-trained model weights**

 All the pretrained classifiers, and pretrained FlowCon models can be found in [link](https://usf.box.com/s/j0dklw5wfqllanzb5v9lc8t8a0uutfpg)

