
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
plt.style.use('tableau-colorblind10')
import pandas as pd
import json
import pprint
import seaborn as sns
import torch

import torchmetrics
from torchmetrics import PrecisionRecallCurve, ROC
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve, confusion_matrix
from torchmetrics.functional.classification import binary_accuracy, multiclass_accuracy, binary_recall, binary_precision, multiclass_recall, multiclass_precision, binary_f1_score, multiclass_f1_score
from torchmetrics.utilities.compute import _auc_compute_without_check, _auc_compute

from utils import get_roc_curve, get_pr_curve, get_confusion_matrix, get_optimal_operating_point

from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score
import statistics
import logging


def bootstrap(y_pred, y_true, n):
    rng_seed = 42
    bootstrapped_scores = []
    bootstrapped_accuracy = []
    n_classes = 1
    if len(y_pred.shape) > 1:
        n_classes = y_pred.shape[1]
    # rng = np.random.RandomState(rng_seed=4)
    for i in range(n):
        # bootstrap by sampling with replacement on the prediction indices
        indices = torch.randint(low=0, high=len(y_true), size=(len(y_true), ))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if len(y_pred.shape) > 1:
            score = multiclass_auroc(y_pred[indices], y_true[indices], num_classes=n_classes, average=None)
        else: 
            score = binary_auroc(y_pred[indices], y_true[indices])

        bootstrapped_scores.append(score)

    if n_classes > 1:
        for i in range(n_classes):
            print('Class ', i)

            sub_array = [x[i] for x in bootstrapped_scores]
            # print(sub_array)
            sorted_scores = np.array(sub_array)
            sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]


            print("n={} Confidence interval for the score: [{:0.3f} - {:0.3}]".format(n,
            confidence_lower, confidence_upper))

            print('MEAN: ', np.mean(sorted_scores))
            print('MEDIAN: ', statistics.median(sorted_scores))

        mean_array = [torch.mean(x) for x in bootstrapped_scores]

        sorted_scores = np.array(mean_array)
        sorted_scores.sort()

        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]


        print("n={} MEAN Confidence interval for the score: {:0.3f}".format(n,
        confidence_lower, confidence_upper))
        
    else:

        sorted_scores = np.array(bootstrapped_scores)
        
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

        print("n={} Confidence interval for the score: [{:0.3f} - {:0.3}]".format(n,
        confidence_lower, confidence_upper))

        print('MEAN: ', np.mean(sorted_scores))
        print('MEDIAN: ', statistics.median(sorted_scores))

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TransMIL', type=str)
    parser.add_argument('--task', default='norm_rest',type=str)
    parser.add_argument('--target_label', default = 1, type=int)
    args = parser.parse_args()
    return args


args = make_parse()

ckpt_dict = {
    'TransMIL': {
        # 'norm_rest': {'version': '893', 'epoch': '166', 'labels':['Normal', 'Disease']},
        'norm_rest': {'version': '804', 'epoch': '30', 'labels':['Normal', 'Disease']},
        'rest_rej': {'version': '63', 'epoch': '14', 'labels': ['Rest', 'Rejection']},
        'norm_rej_rest': {'version': '53', 'epoch': '17', 'labels': ['Normal', 'Rejection', 'Rest']},
    },
    'vit': {
        'norm_rest': {'version': '16', 'epoch': '142', 'labels':['Normal', 'Disease']},
        'rej_rest': {'version': '1', 'epoch': 'last', 'labels': ['Rejection', 'Rest']},
        'norm_rej_rest': {'version': '0', 'epoch': '226', 'labels': ['Normal', 'Rejection', 'Rest']},
    },
    'CLAM': {
        'norm_rest': {'labels':['NORMAL', 'REST']},
        'rej_rest': {'labels': ['REJECTION', 'REST']},
        'norm_rej_rest': {'labels': ['NORMAL', 'REJECTION', 'REST']},
    }
}
def generate_plots(model, task, version=None, epoch=None, labels=None, add_on=0):

    print('-----------------------------------------------------------------------')
    print(model, task, version, epoch, labels, add_on)
    print('-----------------------------------------------------------------------')



    if model == 'CLAM':
        patient_result_csv_path = Path(f'/homeStor1/ylan/workspace/HIA/logs/DeepGraft_Lancet/clam_mb/DEEPGRAFT_CLAMMB_TRAINFULL_{task}/RESULTS/TEST_RESULT_PATIENT_BASED_FULL.csv')
        threshold_csv_path = ''
    else: 
        #TransMIL and ViT
        root_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/{model}/{task}/_{a}_CrossEntropyLoss/lightning_logs/version_{version}/test_epoch_{epoch}'
        # print(root_dir)
        patient_result_csv_path = Path(root_dir) / 'TEST_RESULT_PATIENT.csv'
        # print(patient_result_csv_path)
        threshold_csv_path = f'{root_dir}/val_thresholds.csv'
        thresh_df = pd.read_csv(threshold_csv_path, index_col=False)
        optimal_threshold = thresh_df['patient'].values[0]
        
        # threshold = 
    #####
    ######


    # output_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/results/{model}/'
    output_dir = f'/homeStor1/ylan/DeepGraft_project/DeepGraft_Draft/figures/{model}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    patient_result = pd.read_csv(patient_result_csv_path)
    # pprint.pprint(patient_result)

    probs = np.array(patient_result[labels[int(add_on)]])
    if task == 'norm_rej_rest':
        probs = np.array(patient_result[labels])
    probs = probs.squeeze()
    probs = torch.from_numpy(probs)

        
    # probs = torch.transpose(probs, 0,1).squeeze()
    target = np.array(patient_result.yTrue, dtype=int)
    target = torch.from_numpy(target)

    # res = bootstrap((probs.cpu().numpy(), target.cpu().numpy()), sklearn.metrics.roc_auc_score, confidence_level=0.95, paired=True, vectorized=False)

    # print('bootstrap AUC: ', res)
    # patient_AUC = self.AUROC(patient_score, patient_target.squeeze())

    #swap values for rest_rej for it to align
    if task == 'rest_rej':
        probs = 1 - probs
        target = -1 * (target-1)
        task = 'rej_rest'
    # 
    if add_on == 0 and task != 'norm_rej_rest':
        probs = 1 - probs
        # target = 1 - target
    # if task == 'norm_rej_rest':
    #         optimal_threshold = 1/3
    # else:
    # if model == 'CLAM':
    #     if task == 'norm_rej_rest':
    #         optimal_threshold = 1/3
    #     else: optimal_threshold = 0.5
    # else: 
    if task == 'norm_rej_rest':
        optimal_threshold = 1/3
    else: optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(probs.unsqueeze(0), target.unsqueeze(0))
    if task != 'norm_rej_rest':
        accuracy = binary_accuracy(probs, target, threshold=optimal_threshold)
        recall = binary_recall(probs, target, threshold=optimal_threshold)
        precision = binary_precision(probs, target, threshold=optimal_threshold)
        f1 = binary_f1_score(probs, target, threshold=optimal_threshold)
    else: 
        accuracy = multiclass_accuracy(probs, target, num_classes=3, average=None)
        recall = multiclass_recall(probs, target, num_classes=3, average=None)
        precision = multiclass_precision(probs, target, num_classes=3, average=None)
        f1 = multiclass_f1_score(probs, target, num_classes=3, average=None)


    print(f'Threshold: {optimal_threshold}')
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1: ', f1)

    bootstrap(probs, target, n=1000)


    ######################################################################################
    # Plot
    ######################################################################################


    # probs = 1-probs

    stage='test'
    comment='patient'

    # for i in range(len(labels)):
    pr_plot = get_pr_curve(probs, target, task=task, model=model, target_label=add_on)
    if task != 'norm_rej_rest':
        pr_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_pr.png', dpi=400)
        pr_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_pr.svg', format='svg')
    pr_plot.figure.clf()

    # roc_plot = get_roc_curve(probs, target, task=task, model=model)
    # if task != 'norm_rej_rest':
    #     roc_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_roc.png', dpi=400)
    #     roc_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_roc.svg', format='svg')
    # roc_plot.figure.clf()

    # cm_plot, _ = get_confusion_matrix(probs, target, task=task, optimal_threshold=optimal_threshold)
    # cm_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_cm.png', dpi=400)
    # cm_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_cm.svg', format='svg')
    # cm_plot.figure.clf()

    # plt.close()


if __name__ == '__main__':

    args = make_parse()
    for model in ckpt_dict.keys():
        for task in ckpt_dict[model].keys():
            labels = ckpt_dict[model][task]['labels']
            for i in range(len(labels)):
                add_on = i
                if model == 'TransMIL':
                    a = 'features'
                    version = ckpt_dict[model][task]['version']
                    epoch = ckpt_dict[model][task]['epoch']
                    labels = ckpt_dict[model][task]['labels']
                elif model == 'vit':
                    a = 'vit'
                    version = ckpt_dict[model][task]['version']
                    epoch = ckpt_dict[model][task]['epoch']
                    labels = ckpt_dict[model][task]['labels']
                elif model == 'CLAM':
                    labels = ckpt_dict[model][task]['labels']

                generate_plots(model=model, task=task, version=version, epoch=epoch, labels=labels, add_on=add_on)
        #         break
        #     break
        # break
