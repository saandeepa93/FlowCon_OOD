1. Follow Heatmap experiment framework
2. Explain why FlowCon is better? Density based approach and class specific
3. Penultimate layer only
4. RAF <-> AFF evaluation. What is OOD in face expression?
5. Timeline





. Literature -> Density based latest on CVPR2020
  + [A Simple Unified Framework for Detecting Out-of-Distribution](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)


************************
  + [Boosting Out-of-distribution Detection with
Typical Features](https://proceedings.neurips.cc/paper_files/paper/2022/file/82b0c1b954b6ef9f3cfb664a82b201bb-Paper-Conference.pdf)

  + [Heatmap-based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/WACV2023/papers/Hornauer_Heatmap-Based_Out-of-Distribution_Detection_WACV_2023_paper.pdf)
  + [Beyond AUROC & co. for evaluating
out-of-distribution detection performance](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/papers/Humblot-Renaux_Beyond_AUROC__Co._for_Evaluating_Out-of-Distribution_Detection_Performance_CVPRW_2023_paper.pdf)

  + [Out-of-Distribution Detection with Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d/sun22d.pdf)

python OOD_Generate_Mahalanobis_exp2.py --dataset raf --net_type resnet --gpu 1 --num_classes 7 --batch 64 --net_c 2
python OOD_Regression_Mahalanobis.py --net_type resnet


out_distribution: svhn
 TNR    AUROC  DTACC  AUIN   AUOUT
 93.82  98.38  94.92  93.02  99.79
Input noise: Mahalanobis_0.001

out_distribution: imagenet_resize
 TNR    AUROC  DTACC  AUIN   AUOUT
 93.34  98.33  94.31  95.42  99.49
Input noise: Mahalanobis_0.001

out_distribution: lsun_resize
 TNR    AUROC  DTACC  AUIN   AUOUT
 95.37  98.57  95.39  96.80  99.49
Input noise: Mahalanobis_0.001

###############################
python OOD_Generate_Mahalanobis_exp2.py --dataset raf --net_type effnet --gpu 1 --num_classes 7 --batch 64 --net_c 1


out_distribution: svhn
 TNR    AUROC  DTACC  AUIN   AUOUT 
100.00 100.00  99.80  99.95 100.00
Input noise: Mahalanobis_0.0

out_distribution: imagenet_resize
 TNR    AUROC  DTACC  AUIN   AUOUT 
 99.89  99.86  98.71  99.60  99.96
Input noise: Mahalanobis_0.0

out_distribution: lsun_resize
 TNR    AUROC  DTACC  AUIN   AUOUT 
 99.94  99.87  99.00  99.65  99.96

out_distribution: cifar10
 TNR    AUROC  DTACC  AUIN   AUOUT 
 99.72  99.58  98.08  98.28  99.88










  {'0.002': [{'lsun_resize': {'AUIN': 0.9159968772185179,
                                         'AUOUT': 0.9718827769132757,
                                         'AUROC': 0.9412050521512386,
                                         'DTACC': 0.8951655801825293,
                                         'TNR': 0.5205}},
                        {'imagenet_resize': {'AUIN': 0.7446115293245945,
                                             'AUOUT': 0.924432614947508,
                                             'AUROC': 0.8266008474576271,
                                             'DTACC': 0.7656550195567144,
                                             'TNR': 0.26639999999999997}},
                        {'svhn': {'AUIN': 0.3521282700002417,
                                  'AUOUT': 0.9356399479733257,
                                  'AUROC': 0.6517361993142128,
                                  'DTACC': 0.6172117217681737,
                                  'TNR': 0.15550092194222498}}]}


MAHA
    {'lsun_resize': {'AUIN': 0.14683431702363461,
                    'AUOUT': 0.634156268287239,
                    'AUROC': 0.2199678617992177,
                    'DTACC': 0.5001370273794004,
                    'TNR': 0.016700000000000048}}
{'imagenet_resize': {'AUIN': 0.17565336244423915,
                  'AUOUT': 0.7322783484630648,
                  'AUROC': 0.37478556062581486,
                  'DTACC': 0.5123157105606259,
                  'TNR': 0.06899999999999995}},
  {'svhn': {'AUIN': 0.1357192935527043,
            'AUOUT': 0.9339393085827903,
            'AUROC': 0.6118344015869747,
            'DTACC': 0.5894820856328467,
            'TNR': 0.1849646588813768}},



git remote set-url origin saandeepa.93@gmail.com:saandeepa93/FlowCon_OOD.git
ssh -vT saandeepa.93@gmail.com

ssh-keygen -t ed25519 -C "saandeepa.93@gmail.com"


SHA256:ggc57KKdIdFJSUm8Ol7yuFyY8ZAoTrOJF6q9rKl/W4o



git remote set-url origin git@github.com:saandeepa93/FlowCon_OOD.git