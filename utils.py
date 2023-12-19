
import random
import torch
import argparse
import os 
import numpy as np 

from umap import UMAP
import pandas as pd
import plotly.express as px
import json 
import plotly.graph_objs as go


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score













def get_curve(known, novel, stype):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
    fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
    tp[stype][0], fp[stype][0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[stype][l+1:] = tp[stype][l]
            fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
            fp[stype][l+1:] = fp[stype][l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[stype][l+1] = tp[stype][l]
                fp[stype][l+1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l+1] = tp[stype][l] - 1
                fp[stype][l+1] = fp[stype][l]
    tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
    tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def metric(known, novel, stype = 'flowcon', verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(known, novel, stype)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    if verbose:
        print('{stype:5s} '.format(stype=stype), end='')
    results[stype] = dict()
    
    # TNR
    mtype = 'TNR'
    results[stype][mtype] = tnr_at_tpr95[stype]
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
    
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
    fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
    results[stype][mtype] = -np.trapz(1.-fpr, tpr)
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
    
    # DTACC
    mtype = 'DTACC'
    results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
    
    # AUIN
    mtype = 'AUIN'
    denom = tp[stype]+fp[stype]
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
    results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
    results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        print('')

    return results

def make_roc(confidence, labels, nth):
    min_thresh = np.amin(confidence)
    max_thresh = np.amax(confidence)

    thresholds = np.linspace(max_thresh, min_thresh, nth)
    # thresholds = -np.sort(-np.unique(confidence))
    thresholds = np.append(thresholds, min_thresh - 1e-3)
    thresholds = np.insert(thresholds, 0, max_thresh + 1e-3)

    fpr = np.empty(len(thresholds))
    tpr = np.empty(len(thresholds))
    precision = np.empty(len(thresholds))


    for i, th in zip(range(len(thresholds)), thresholds):
        tp = float(confidence[(labels == True) & (confidence >= th)].shape[0])
        tn = float(confidence[(labels != True) & (confidence < th)].shape[0])
        fp = float(confidence[(labels != True) & (confidence >= th)].shape[0])
        fn = float(confidence[(labels == True) & (confidence < th)].shape[0])


        fpr[i] = fp / (tn + fp) #FP from R
        tpr[i] = tp / (tp + fn) #TP from P

        if tp != 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0

    recall = tpr
    auroc = np.trapz(tpr, fpr)
    auprre = np.trapz(precision, recall)
    return fpr, tpr, auroc, precision, recall, auprre

def mkdir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True


def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
  parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
  args = parser.parse_args()
  return args

def get_metrics(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  error_rate = 1 - acc
  return round(acc, 3), error_rate




def get_metrics_ood(label, score, invert_score=False):
    results_dict = {}
    if invert_score:
        score = score - score.max()
        score = np.abs(score)

    error = 1 - label
    rocauc = roc_auc_score(label, score)

    aupr_success = average_precision_score(label, score)
    aupr_errors = average_precision_score(error, (1 - score))

    # calculate fpr @ 95% tpr
    fpr = 0
    eval_range = np.arange(score.min(), score.max(), (score.max() - score.min()) / 10000)
    for i, delta in enumerate(eval_range):
        tpr = len(score[(label == 1) & (score >= delta)]) / len(score[(label == 1)])
        if 0.9505 >= tpr >= 0.9495:
            fpr = len(score[(error == 1) & (score >= delta)]) / len(score[(error == 1)])
            break

    results_dict["rocauc"] = rocauc
    results_dict["aupr_success"] = aupr_success
    results_dict["aupr_error"] = aupr_errors
    results_dict["fpr"] = fpr
    return results_dict



def plot_umap(cfg, X_lst_un, y_lst, name, dim, mode, labels_in_ood=None):
  b = X_lst_un.size(0)
  X_lst = UMAP(n_components=dim, random_state=0, init='random').fit_transform(X_lst_un.view(b, -1))

  if labels_in_ood is None:
    y_lst_label = [str(i) for i in y_lst.detach().numpy()]
  else:
     y_lst_label = [str(i) for i in labels_in_ood.detach().numpy()]


  if dim == 3:
    df = pd.DataFrame(X_lst, columns=["x", "y", "z"])
  else:
    df = pd.DataFrame(X_lst, columns=["x", "y"])
  df_color = pd.DataFrame(y_lst_label, columns=["class"])
  df = df.join(df_color)
  if dim == 3:
    fig = px.scatter_3d(df, x='x', y='y', z='z',color='class', title=f"{name}", \
     )
  else:
    fig = px.scatter(df, x='x', y='y',color='class', title=f"{name}", \
    )
  
  fig.update_traces(marker=dict(size=7))
  fig.update_layout(title=None,  margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
  fig.update_layout(
      legend=dict(
          yanchor="top",
          y=0.99,
          xanchor="right",
          x=0.90,
          font = dict(
              family="Courier",
              size=24,
              color='black'
          ),
            
        )
      )
  
  dest_path = os.path.join("./data/umaps/", name)

  mkdir(dest_path)
  fig.write_html(os.path.join(dest_path, f"{dim}d_{mode}_new.html"))
  fig.write_image(os.path.join(dest_path, f"{dim}d_{mode}_new.png"), width=1745, height=840, scale=1)
