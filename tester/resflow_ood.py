"""
Out of distribution Regression based on the code of Kimin Lee (Sun Oct 21 2018)
"""
from __future__ import print_function

import sys
sys.path.append('.')
import numpy as np
import lib_regression_res as lib_regression
import argparse

from sklearn.linear_model import LogisticRegressionCV

from utils import *

parser = argparse.ArgumentParser(description='PyTorch code: Residual flow regression')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--config', type=str, default='default', help='configuration to load')
args = parser.parse_args()
print(args)

def main():
    # initial setup

    dataset_list = ['cifar100']

    score_list = ['0.0', '0.01', '0.005', '0.002', '0.0014', '0.001', '0.0005']
    
    main_score_dict = {}
    # train and measure the performance of residual flow detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        main_score_dict[dataset] = {}
        print('In-distribution: ', dataset)
        outf = './data/baselines/resflow_feat_list/' + args.net_type + '_' + dataset + 'RealNVP_magnitude' + '/'
        # out_list = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
        # out_list = ['cifar10']
        out_list = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365', 'cifar10', 'cifar100']

        out_dict = {}
        # out_list = ['lsun-r', 'lsun-c', 'isun']
        if dataset == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']

        if dataset == 'cifar100':
            num_classes = 100
        else:
            num_classes = 10

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            main_score_dict[dataset][out] = {}
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                
                total_X, total_Y = lib_regression.load_characteristics_RealNVP(score, dataset, out, outf, args.net_type, num_classes)
                X_val, Y_val, X_test, Y_test, partition = lib_regression.block_split_RealNVP(total_X, total_Y, out)

                out_dict[out] = partition
                X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
                Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
                lr = LogisticRegressionCV(n_jobs=1).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]


                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                
                results, results_ours = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_tnr < results_ours['aupr_success']:
                    best_tnr = results_ours['aupr_success']
                    best_index = score
                    best_result, best_result_ours = lib_regression.detection_performance(lr, X_test, Y_test, outf)

                    y_pred = lr.predict_proba(X_test)[:, 1]
                    np.save(f"./data/baselines/resflow_feat_list/bpds/{args.net_type}_{dataset}_{out}_test", y_pred)

                    print(best_result_ours)
                    main_score_dict[dataset][out][score] = best_result_ours 
                # if best_tnr < results['TMP']['TNR']:
                #     best_tnr = results['TMP']['TNR']
                #     best_index = score
                #     best_result, best_result_ours = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result_ours)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)

    e()
    res_path = os.path.join(f"./data/baselines/resflow_feat_list/results")
    mkdir(res_path)
    with open(f'{res_path}/{args.net_type}_{dataset_list[0]}_all.json', 'w') as fp:
        json.dump(main_score_dict, fp, indent=4)

    # print the results
    count_in = 0
    # mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    mtypes = ['rocauc', 'aupr_success', 'aupr_error', 'fpr']
    avg_result_dict = {"rocauc": 0, "aupr_success": 0, "aupr_error": 0, "fpr": 0}
    auroc_lst, auprs_lst, aupre_lst, fpr_lst = [], [], [], []
    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        # out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        # out_list = ['lsun-r', 'lsun-c', 'isun', 'svhn', 'textures', 'places365']
        if dataset_list[count_in] == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        count_out = 0

        for results in in_list:
            
            print('out_distribution: '+ out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            
            print('\n{val:6.2f}'.format(val=100.*results["rocauc"]), end='')
            print(' {val:6.2f}'.format(val=100.*results['aupr_success']), end='')
            print(' {val:6.2f}'.format(val=100.*results['aupr_error']), end='')
            print(' {val:6.2f}'.format(val=100.*results['fpr']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])

            auroc_lst.append(results["rocauc"])
            auprs_lst.append(results["aupr_success"])
            aupre_lst.append(results["aupr_error"])
            fpr_lst.append(results["fpr"])

            # avg_result_dict["rocauc"] += results["rocauc"]
            # avg_result_dict["aupr_success"] += results["aupr_success"]
            # avg_result_dict["aupr_error"] += results["aupr_error"]
            # avg_result_dict["fpr"] += results["fpr"]

            print('')
            count_out += 1
        count_in += 1
        avg_result_dict['rocauc'] = sum(auroc_lst)/len(auroc_lst)
        avg_result_dict['aupr_success'] = sum(auprs_lst)/len(auprs_lst)
        avg_result_dict['aupr_error'] = sum(aupre_lst)/len(aupre_lst)
        avg_result_dict['fpr'] = sum(fpr_lst)/len(fpr_lst)

    # avg_result_dict["rocauc"] = avg_result_dict["rocauc"] / len(out_list)
    # avg_result_dict["aupr_success"] = avg_result_dict["aupr_success"] / len(out_list)
    # avg_result_dict["aupr_error"] = avg_result_dict["aupr_error"] / len(out_list)
    # avg_result_dict["fpr"] = avg_result_dict["fpr"] / len(out_list)
        res_path = os.path.join(f"./data/baselines/resflow_feat_list/results")
        mkdir(res_path)
        with open(f'{res_path}/{args.net_type}_{dataset_list[0]}_near.json', 'w') as fp:
            json.dump(avg_result_dict, fp, indent=4)

if __name__ == '__main__':
    main()