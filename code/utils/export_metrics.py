
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
from sklearn.metrics import PrecisionRecallDisplay

from utils import get_roc_curve, get_pr_curve, get_confusion_matrix, get_optimal_operating_point, get_roc_curve_2, get_pr_curve_2

from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score
import statistics
import logging
import scienceplots
import itertools


def bootstrap(y_pred, y_true, n, class_specific=False):
    rng_seed = 42
    bootstrapped_auroc = []
    bootstrapped_accuracy = []
    bootstrapped_precision = []
    bootstrapped_recall = []
    bootstrapped_f1 = []
    n_classes = 1
    average = 'macro'
    if class_specific: 
        average = None

    if len(y_pred.shape) > 1:
        n_classes = y_pred.shape[1]
    # rng = np.random.RandomState(rng_seed=4)
    for i in range(n):
        # bootstrap by sampling with replacement on the prediction indices
        indices = torch.randint(low=0, high=len(y_true), size=(len(y_true), ))
        # print(indices)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if len(y_pred.shape) > 1:
            auroc = multiclass_auroc(y_pred[indices], y_true[indices], num_classes=n_classes, average=average)
            accuracy = multiclass_accuracy(y_pred[indices], y_true[indices], num_classes=n_classes, average=average)
            precision = multiclass_precision(y_pred[indices], y_true[indices], num_classes=n_classes, average=average)
            recall = multiclass_recall(y_pred[indices], y_true[indices], num_classes=n_classes, average=average)
            f1 = multiclass_f1_score(y_pred[indices], y_true[indices], num_classes=n_classes, average=average)
        else: 
            score = binary_auroc(y_pred[indices], y_true[indices])
 

        bootstrapped_auroc.append(auroc)
        bootstrapped_accuracy.append(accuracy)
        bootstrapped_precision.append(precision)
        bootstrapped_recall.append(recall)
        bootstrapped_f1.append(f1)


    class_ci = np.zeros([n_classes])
    if n_classes > 1:
        
        ##################
        # Class specific #
        ##################
    #     for i in range(n_classes):
    #         print('Class ', i)
    #         sorted_auroc = np.sort(np.array([x[i] for x in bootstrapped_auroc]))
    #         sorted_accuracy = np.sort(np.array([x[i] for x in bootstrapped_accuracy]))
    #         sorted_precision = np.sort(np.array([x[i] for x in bootstrapped_precision]))
    #         sorted_recall = np.sort(np.array([x[i] for x in bootstrapped_recall]))
    #         sorted_f1 = np.sort(np.array([x[i] for x in bootstrapped_f1]))
    #         # sorted_scores.sort()

    # #     # Computing the lower and upper bound of the 90% confidence interval
    # #     # You can change the bounds percentiles to 0.025 and 0.975 to get
    # #     # a 95% confidence interval instead.
    # #         auroc_ci = (sorted_auroc[int(0.025 * len(sorted_auroc))], sorted_auroc[int(0.975 * len(sorted_auroc))], np.mean(sorted_auroc))
    # #         accuracy_ci = (sorted_accuracy[int(0.025 * len(sorted_accuracy))], sorted_accuracy[int(0.975 * len(sorted_accuracy))], np.mean(sorted_accuracy))
    #         precision_ci = (sorted_precision[int(0.025 * len(sorted_precision))], sorted_precision[int(0.975 * len(sorted_precision))], np.mean(sorted_precision))
            # print( np.mean(sorted_precision))
    #         recall_ci = (sorted_recall[int(0.025 * len(sorted_recall))], sorted_recall[int(0.975 * len(sorted_recall))], np.mean(sorted_recall))
    #         f1_ci = (sorted_f1[int(0.025 * len(sorted_f1))], sorted_f1[int(0.975 * len(sorted_f1))], np.mean(sorted_f1))
            
            # confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            # confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            
        if not class_specific:

            # mean_auroc = [torch.mean(x) for x in bootstrapped_auroc]
            sorted_auroc = np.sort(np.array(bootstrapped_auroc))
            sorted_accuracy = np.sort(np.array(bootstrapped_accuracy))
            sorted_precision = np.sort(np.array(bootstrapped_precision))
            sorted_recall = np.sort(np.array(bootstrapped_recall))
            sorted_f1 = np.sort(np.array(bootstrapped_f1))

            auroc_ci = (sorted_auroc[int(0.025 * len(sorted_auroc))], sorted_auroc[int(0.975 * len(sorted_auroc))], np.mean(sorted_auroc))
            accuracy_ci = (sorted_accuracy[int(0.025 * len(sorted_accuracy))], sorted_accuracy[int(0.975 * len(sorted_accuracy))], np.mean(sorted_accuracy))
            precision_ci = (sorted_precision[int(0.025 * len(sorted_precision))], sorted_precision[int(0.975 * len(sorted_precision))], np.mean(sorted_precision))
            recall_ci = (sorted_recall[int(0.025 * len(sorted_recall))], sorted_recall[int(0.975 * len(sorted_recall))], np.mean(sorted_recall))
            f1_ci = (sorted_f1[int(0.025 * len(sorted_f1))], sorted_f1[int(0.975 * len(sorted_f1))], np.mean(sorted_f1))


        # sorted_scores = np.array(mean_array)
        # sorted_scores.sort()

        # confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        # confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]


        # print("n={} MEAN Confidence interval for the score: {:0.3f}".format(n,
        # confidence_lower, confidence_upper))
        
    else:

        sorted_scores = np.array(bootstrapped_scores)
        
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

        print("n={} Confidence interval for the score: [{:0.3f} - {:0.3}]".format(n,
        confidence_lower, confidence_upper))

        print('MEAN: ', np.mean(sorted_scores))
        print('MEDIAN: ', statistics.median(sorted_scores))
    
    # return (sorted_auroc, sorted_accuracy, sorted_precision, sorted_recall, sorted_f1), (auroc_ci, accuracy_ci, precision_ci, recall_ci, f1_ci)

def make_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', default='RCC', type=str)
    parser.add_argument('--model', default='TransMIL', type=str)
    # parser.add_argument('--task', default='norm_rest',type=str)
    parser.add_argument('--target_label', default = 1, type=int)
    args = parser.parse_args()
    return args


args = make_parse()

deepgraft_dict = {
    'TransMIL': {
        # 'norm_rest': {'version': '893', 'epoch': '166', 'labels':['Normal', 'Disease']},
        'norm_rest': {'version': '804', 'epoch': '30', 'labels':['Normal', 'Disease']},
        'rej_rest': {'version': '63', 'epoch': '14', 'labels': ['Rest', 'Rejection']},
        'norm_rej_rest': {'version': '53', 'epoch': '17', 'labels': ['Normal', 'Rejection', 'Rest']},
        # 'norm_rej_rest': {'version': '53', 'epoch': '25', 'labels': ['Normal', 'Rejection', 'Rest']},
    },
    'ViT': {
        'norm_rest': {'version': '16', 'epoch': '142', 'labels':['Normal', 'Disease']},
        'rej_rest': {'version': '1', 'epoch': 'last', 'labels': ['Rejection', 'Rest']},
        'norm_rej_rest': {'version': '0', 'epoch': '226', 'labels': ['Normal', 'Rejection', 'Rest']},
    },
    'CLAM': {
        'norm_rest': {'labels':['NORMAL', 'REST']},
        'rej_rest': {'labels': ['REJECTION', 'REST']},
        'norm_rej_rest': {'labels': ['NORMAL', 'REJECTION', 'REST']},
    },
    'InceptionNet': {
        'norm_rest': {'labels':['Normal', 'Disease']},
        'rej_rest': {'labels': ['Rest', 'Rejection']},
        'norm_rej_rest': {'labels': ['Normal', 'Rejection', 'Rejection']},
    }
}

rcc_dict = {
    'TransMIL':{
        'big_three': {'version': '14', 'epoch':'244', 'labels':['ccRCC', 'papRCC', 'chRCC']}
    },
    'CLAM':{
        # 'big_three': {'labels':['ccRCC', 'papRCC', 'chRCC']}
        'big_three': {'labels':['CCRCC', 'CHRCC', 'PAPRCC']}
    },
    'InceptionNet': {
        'big_three': {'version': '2', 'epoch': '79', 'labels':['ccRCC', 'papRCC', 'chRCC']}
    },
    'ViT': {
        'big_three': {'version': '4', 'epoch': '27', 'labels': ['ccRCC', 'papRCC', 'chRCC']}
    },
}

def get_data(project, model, task, version=None, epoch=None, labels=None, add_on=0): #, mode, ax

    print('-----------------------------------------------------------------------')
    print(project, model, task, f'v{version}', f'e{epoch}', labels, add_on)
    print('-----------------------------------------------------------------------')


    if project == 'DeepGraft':
        if model == 'CLAM':
            patient_result_csv_path = Path(f'/homeStor1/ylan/workspace/HIA/logs/DeepGraft_Lancet/clam_mb/DEEPGRAFT_CLAMMB_TRAINFULL_{task}/RESULTS/TEST_RESULT_PATIENT_BASED_FULL.csv')
            threshold_csv_path = ''
        elif model == 'InceptionNet':
            if task == 'rej_rest':
                    task = 'rest_rej'
            patient_result_csv_path = Path(f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/inception_results/{task}.csv')
            
            # test/inception_results/rest_rej.csv
            # Path(f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/inception_results/{task}.csv')
        else: 
            #TransMIL and ViT
            if model == 'TransMIL':
                a = 'features'
                if task == 'rej_rest':
                    task = 'rest_rej'
                root_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/{model}/{task}/_features_CrossEntropyLoss/lightning_logs/version_{version}/test_epoch_{epoch}_best'
                if not Path(root_dir).exists():
                    root_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/{model}/{task}/_features_CrossEntropyLoss/lightning_logs/version_{version}/test_epoch_{epoch}'

                
            elif model == 'ViT':
                a = 'vit'
                root_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/vit/{task}/_vit_CrossEntropyLoss/lightning_logs/version_{version}/test_epoch_{epoch}'
            # print(root_dir)
            patient_result_csv_path = Path(root_dir) / 'TEST_RESULT_PATIENT.csv'
            # print(patient_result_csv_path)
            threshold_csv_path = f'{root_dir}/val_thresholds.csv'
            thresh_df = pd.read_csv(threshold_csv_path, index_col=False)
            optimal_threshold = thresh_df['patient'].values[0]
    elif project == 'RCC':
        if model == 'CLAM':
            patient_result_csv_path = Path(f'/homeStor1/ylan/workspace/HIA/logs/RCC_Lancet/clam_mb/RCC_CLAMMB_TRAINFULL_{task}/RESULTS/TEST_RESULT_PATIENT_BASED_FULL.csv')
            threshold_csv_path = ''
        elif model == 'AttMIL':
            patient_result_csv_path = Path(f'/homeStor1/ylan/workspace/HIA/logs/RCC_Lancet/attmil/RCC_ATTMIL_TRAINFULL_{task}/RESULTS/TEST_RESULT_PATIENT_BASED_FULL.csv')
        # elif model == 'Inception':
        #     # if task == 'rej_rest':
        #     #         task = 'rest_rej'
        #     # patient_result_csv_path = Path(f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/inception_results/{task}.csv')
            
        #     # test/inception_results/rest_rej.csv
        #     # Path(f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/inception_results/{task}.csv')
        #     a = 'inception'
        #     root_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/{project}/inception/{task}/_{a}_CrossEntropyLoss/lightning_logs/version_{version}/test_epoch_{epoch}'
            
        else: 
            #TransMIL and ViT
            if model == 'TransMIL':
                a = 'features'
                model_name = model
                
            elif model == 'ViT':
                a = 'vit'
                model_name = 'vit'
            elif model == 'InceptionNet':
                a = 'inception'
                model_name = 'inception'
            root_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/{project}/{model_name}/{task}/_{a}_CrossEntropyLoss/lightning_logs/version_{version}/test_epoch_{epoch}'
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
    

    patient_result = pd.read_csv(patient_result_csv_path)

    # probs = np.array(patient_result[labels[1]])
    
    if task in ['big_three', 'norm_rej_rest']:
    
        # print(labels)
        probs = np.array(patient_result[labels])
    else:
        # print(patient_result[labels[0]])
        probs = np.array(patient_result[labels[int(add_on)]])



    probs = probs.squeeze()
    probs = torch.from_numpy(probs)

    target = np.array(patient_result.yTrue, dtype=int)
    target = torch.from_numpy(target)


    #swap values for rest_rej for it to align
    if task == 'rest_rej':
        probs = 1 - probs
        target = -1 * (target-1)
        task = 'rej_rest'
    # 
    # print(probs)
    # print(target)
    # if add_on == 0 and task != 'norm_rej_rest':
    #     # probs = 1 - probs
    #     target = 1 - target
    #     print(probs)
    #     print(target)

    # if task == ('norm_rej_rest' or 'big_three'):
    # print(probs.shape)

    if task in ['big_three', 'norm_rej_rest']:
        optimal_threshold = 1/3
        accuracy = multiclass_accuracy(probs, target, num_classes=3, average=None)
        recall = multiclass_recall(probs, target, num_classes=3, average=None)
        precision = multiclass_precision(probs, target, num_classes=3, average=None)
        f1 = multiclass_f1_score(probs, target, num_classes=3, average=None)
        auroc = multiclass_auroc(probs, target, num_classes=3, average=None)

        # print(task)
    else:
        optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(probs.unsqueeze(0), target.unsqueeze(0))
        accuracy = binary_accuracy(probs, target, threshold=optimal_threshold)
        recall = binary_recall(probs, target, threshold=optimal_threshold)
        precision = binary_precision(probs, target, threshold=optimal_threshold)
        f1 = binary_f1_score(probs, target, threshold=optimal_threshold)
        positive_probs = probs[target == 1]
        # print(positive_probs)
        # print(positive_probs.mean())
        # for i in positive_probs:
            # if i < 0.5:
                # print(i)

    # if task != ('norm_rej_rest' and 'big_three'):
    #     accuracy = binary_accuracy(probs, target, threshold=optimal_threshold)
    #     recall = binary_recall(probs, target, threshold=optimal_threshold)
    #     precision = binary_precision(probs, target, threshold=optimal_threshold)
    #     f1 = binary_f1_score(probs, target, threshold=optimal_threshold)
    #     positive_probs = probs[target == 1]
    #     print(positive_probs)
    #     print(positive_probs.mean())
    #     for i in positive_probs:
    #         if i < 0.5:
    #             print(i)


    # else: 
    #     accuracy = multiclass_accuracy(probs, target, num_classes=3, average=None)
    #     recall = multiclass_recall(probs, target, num_classes=3, average=None)
    #     precision = multiclass_precision(probs, target, num_classes=3, average=None)
    #     f1 = multiclass_f1_score(probs, target, num_classes=3, average=None)



    print(f'Threshold: {optimal_threshold}')
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1: ', f1)
    print('AUROC: ', auroc)
    # print(probs, target)
    # bootstrap(probs, target, n=1000)

    return probs, target, optimal_threshold

col_dict = {'norm_rest': ['Normal', 'Disease'],
            'rej_rest': ['Rejection', 'Other'],
            'norm_rej_rest': ['Normal', 'Rejection', 'Viral+Other'],
            'big_three': ['ccRCC', 'papRCC', 'chRCC'],
            }
row_list = ['TransMIL', 'CLAM', 'ViT', 'InceptionNet']

def plot_separately(project, models, mode, tasks, output_path):
    
    # models = ['TransMIL', 'CLAM', 'vit', 'Inception'] #
    # models = ['TransMIL'] #'TransMIL', 
    if project == 'DeepGraft':
        model_dict = deepgraft_dict
    elif project == 'RCC':
        model_dict = rcc_dict
    
    # for m in mode:
    for task in tasks: #deepgraft_dict[model].keys()
        for model in models:
            output_dir = Path(output_path) / model
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            labels = model_dict[model][task]['labels']
            if model == 'TransMIL':
                version = model_dict[model][task]['version']
                epoch = model_dict[model][task]['epoch']
                labels = model_dict[model][task]['labels']
                m_name='transmil'
            elif model == 'ViT':
                m_name='vit'
                version = model_dict[model][task]['version']
                epoch = model_dict[model][task]['epoch']
                labels = model_dict[model][task]['labels']
                
            elif model == 'CLAM':
                m_name='clam'
                labels = model_dict[model][task]['labels']
                version = 0
                epoch = 0
            elif model == 'InceptionNet':
                labels = model_dict[model][task]['labels']
                m_name='inception'

                if project == 'RCC':
                    version = model_dict[model][task]['version']
                    epoch = model_dict[model][task]['epoch']
                else: 
                    version = 0
                    epoch = 0
            if task in ['big_three', 'norm_rej_rest']:
                # print(len(labels))
                for i in range(len(labels)):
                    add_on = i
                    
                    ###################################
                    # Generate plots separately
                    ###################################
                    with plt.style.context(['science', 'nature']):


                        fig, ax = plt.subplots(figsize=(10,10))
                        probs, target, optimal_threshold = get_data(project=project, model=model, task=task, version=version, epoch=epoch, labels=labels, add_on=add_on)
                        # if add_on == 0:
                        #         target = 1-target
                        # print(labels[add_on])
                        for m in mode:
                            if m == 'roc':
                                get_roc_curve_2(probs, target, task=task, ax=ax, target_label=add_on, target_class = labels[add_on])
                            elif m == 'pr':
                                get_pr_curve_2(probs, target, task=task, target_label=add_on, ax=ax, target_class = labels[add_on])

                        #     # get_pr_curve(probs, target, task, model=model, ax=ax, target_label=add_on)
                        # # fig.savefig(f'{output_dir}/{model}_{task}_{add_on}_{m}.png', dpi=400)
                        # # fig.savefig(f'{output_dir}/{model}_{task}_{add_on}_{m}.svg', format='svg')
                        # # fig.clf()
                        #     fig.savefig(f'{output_dir}/{model}_{task}_{add_on}_{m}.png', dpi=400)
                        #     fig.savefig(f'{output_dir}/{model}_{task}_{add_on}_{m}.svg', format='svg')
                        #     fig.clf()

            else: 
                fig, ax = plt.subplots(figsize=(10,10))
                # print('add_on: ', add_on)
                probs, target, optimal_threshold = get_data(project=project, model=model, task=task, version=version, epoch=epoch, labels=labels, add_on=0)
                # print(probs)
                # print(target)
                for m in mode:
                    if m == 'roc':
                        get_roc_curve_2(probs, target, task=task, ax=ax, target_label=0)

                    elif m == 'pr':
                        get_pr_curve_2(probs, target, task=task, target_label=0, ax=ax)
            fig.savefig(f'{output_dir}/{model}_{task}_{m}.png', dpi=400)
            # fig.savefig(f'{output_dir}/{model}_{task}_{m}.svg', format='svg')
            fig.clf()
                

def plot_combined(project, mode, tasks, output_path):
    
    if project == 'DeepGraft':
        model_dict = deepgraft_dict
    elif project == 'RCC':
        model_dict = rcc_dict
    # models = ['TransMIL', 'CLAM', 'vit', 'Inception'] #
    models = ['TransMIL']

    for m in mode:
        for task in tasks: #deepgraft_dict[model].keys()
            # output_dir = f'/homeStor1/ylan/DeepGraft_project/DeepGraft_Draft/figures/{model}'
            output_dir = Path(output_path)
            # output_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/debug/'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plt.figure(1)
            if task in ['big_three', 'norm_rej_rest']:
                n_cols = 1
                n_rows = 4
            else:
                n_cols = 2
                n_rows = 4
            with plt.style.context(['science', 'nature']):
                # fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10*n_cols,10*n_rows), layout='constrained')
                cols = col_dict[task]
                pad=15
                for r in range(len(row_list)):
                    fig, axs = plt.subplots(figsize=(10,10), layout='constrained')
                    model = row_list[r]
                    # if model == 'ViT':
                    #     m_name = 'vit' 
                    # elif model == 'InceptionNet':
                    #     m_name = 'Inception'
                    # else:
                    #     m_name = model

                    for c in range(len(cols)):
                        # print(model)
                        if model == 'TransMIL':
                            version = model_dict[model][task]['version']
                            epoch = model_dict[model][task]['epoch']
                            labels = model_dict[model][task]['labels']
                            m_name='transmil'
                        elif model == 'ViT':
                            m_name='vit'
                            version = model_dict[model][task]['version']
                            epoch = model_dict[model][task]['epoch']
                            labels = model_dict[model][task]['labels']
                            
                        elif model == 'CLAM':
                            m_name='clam'
                            labels = model_dict[model][task]['labels']
                            version = 0
                            epoch = 0
                        elif model == 'InceptionNet':
                            labels = model_dict[model][task]['labels']
                            m_name='inception'

                            if project == 'RCC':
                                version = model_dict[model][task]['version']
                                epoch = model_dict[model][task]['epoch']
                            else: 
                                version = 0
                                epoch = 0
                            # version = 0
                            # epoch = 0
                        # print(model)
                        # print(version)
                        # print(epoch)
                        probs, target, optimal_threshold = get_data(project=project, model=model, task=task, version=version, epoch=epoch, labels=labels, add_on=c)
                        # if c == 0 and (task != 'norm_rej_rest' or task != 'big_three'): # for binary tasks!!
                        (sorted_auroc, sorted_accuracy, sorted_precision, sorted_recall, sorted_f1), (auroc_ci, accuracy_ci, precision_ci, recall_ci, f1_ci)= bootstrap(probs, target, n=1000)
                        #     target = 1-target
                        # print(labels[c])
                        if m == 'roc':
                            mean_auroc = auroc_ci[-1]
                            
                            get_roc_curve_2(probs, target, mean_auroc, task=task, ax=axs, target_label=c, target_class=labels[c])
                        elif m == 'pr':
                            mean_precision = precision_ci[-1]
                            get_pr_curve_2(probs, target, mean_precision, task=task, ax=axs, target_label=c, target_class=labels[c])
                        # elif m == 'cm':

                        #     get_confusion_matrix(probs, target, task, optimal_threshold, ax=axs[r,c])

                        # sub_title = f'{chr(r + 97).upper()}{c+1}'
                        # axs[r].set_title(sub_title, x=-0.1, y=1.05, fontsize=50)
                        # if c != 0:
                        #     axs[r].set_ylabel('')
                        # if r != len(models)-1:
                        #     axs[r].set_xlabel('')

                # for ax, col in zip(axs[0], cols):
                #     ax.annotate(col, xy=(0.5, 1), xytext=(0, 50),
                #                 xycoords='axes fraction', textcoords='offset points',
                #                 fontsize=60, ha='center', va='baseline')

                # for ax, row in zip(axs[1], row_list):
                #     ax.set_ylabel(row, rotation=90, fontsize=30)
                # for ax, row in zip(axs[:,0], row_list):

                #     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                #                 xycoords=ax.yaxis.label, textcoords='offset points',
                #                 fontsize=60,ha='center', va='center', rotation=90) # 
                    # plt.title(model, fontsize=40)
                    fig.savefig(f'{output_dir}/{project}_{model}_{m}.png', dpi=400)
                    fig.savefig(f'{output_dir}/{project}_{model}_{m}.svg', format='svg')
                    fig.clf()

def plot_combined_cm(project, tasks, output_path):
    
    # print(mode)
    # assert mode in ['roc', 'pr']
    if project == 'DeepGraft':
        model_dict = deepgraft_dict
    elif project == 'RCC':
        model_dict = rcc_dict
    
    models = ['TransMIL', 'CLAM', 'vit', 'Inception'] #
    task_list = ['Normal vs Disease', 'Rejection vs Other', 'Normal vs Rejection vs Other']

    # for task in tasks: #deepgraft_dict[model].keys()
    # output_dir = f'/homeStor1/ylan/DeepGraft_project/DeepGraft_Draft/figures/{model}'
    output_dir = Path(output_path)
    # output_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/debug/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(1)
    n_cols = 3
    n_rows = 4
    with plt.style.context(['science', 'nature']):
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10*n_cols,10*n_rows), layout='constrained')
        # cols = col_dict[task]
        cols = col_dict.keys()
        pad=15
        for r in range(len(row_list)):
            model = models[r]
            if model == 'ViT':
                m_name = 'vit'
            elif model == 'Inception':
                m_name = 'InceptionNet'
            else:
                m_name = model
            for c in range(len(tasks)): #deepgraft_dict[model].keys()
                task = tasks[c]
                if model == 'TransMIL':
                    a = 'features'
                    version = model_dict[model][task]['version']
                    epoch = model_dict[model][task]['epoch']
                    labels = model_dict[model][task]['labels']
                elif model == 'vit':
                    a = 'vit'
                    version = model_dict[model][task]['version']
                    epoch = model_dict[model][task]['epoch']
                    labels = model_dict[model][task]['labels']
                    print(version)
                    print(labels)
                elif model == 'CLAM':
                    labels = model_dict[model][task]['labels']
                    version = 0
                    epoch = 0
                elif model == 'Inception':
                    labels = model_dict[model][task]['labels']
                    version = 0
                    epoch = 0
            # for c in indexes:

            # for c in range(len(cols)): #deepgraft_dict[model].keys()
                # col_name = col_dict[col_dict.keys()[c]]
                

                probs, target, optimal_threshold = get_data(project=project, model=model, task=task, version=version, epoch=epoch, labels=labels, add_on=1)
            # if c == 0 and task != 'norm_rej_rest':
                # target = 1-target
            # if m == 'roc':
            #     get_roc_curve_2(probs, target, task=task, ax=axs[r,c])
            # elif m == 'pr':
                
            #     get_pr_curve_2(probs, target, task=task, ax=axs[r,c], target_label=c)
            # elif m == 'cm':

                get_confusion_matrix(probs, target, task, optimal_threshold, ax=axs[r,c])

                sub_title = f'{chr(r + 97).upper()}{c+1}'
                axs[r,c].set_title(sub_title, x=-0.1, y=1.05, fontsize=50)
                if c != 0:
                    axs[r,c].set_ylabel('')
                if r != len(models)-1:
                    axs[r,c].set_xlabel('')

        for ax, col in zip(axs[0], col_dict.keys()):
            col_name = ' vs '.join(col_dict[col])
            ax.annotate(col_name, xy=(0.5, 1), xytext=(0, 50),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=50, ha='center', va='baseline')

        # for ax, row in zip(axs[1], row_list):
        #     ax.set_ylabel(row, rotation=90, fontsize=30)
        for ax, row in zip(axs[:,0], row_list):

            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        fontsize=60,ha='center', va='center', rotation=90) # 

        fig.savefig(f'{output_dir}/combined_cm.png', dpi=100)
        # fig.savefig(f'{output_dir}/{task}_{m}.svg', format='svg')
        fig.clf()
        
        
def get_csv(project, tasks, output_path):
    
    if project == 'DeepGraft':
        model_dict = deepgraft_dict
    elif project == 'RCC':
        model_dict = rcc_dict
    model_list = ['TransMIL', 'CLAM', 'ViT', 'InceptionNet'] #
    # model_list = ['TransMIL']

    for task in tasks: 
        complete_ci_list = []
        for model in model_list:
            if model == 'TransMIL':
                version = model_dict[model][task]['version']
                epoch = model_dict[model][task]['epoch']
                labels = model_dict[model][task]['labels']
                m_name='transmil'
            elif model == 'ViT':
                m_name='vit'
                version = model_dict[model][task]['version']
                epoch = model_dict[model][task]['epoch']
                labels = model_dict[model][task]['labels']
                
            elif model == 'CLAM':
                m_name='clam'
                labels = model_dict[model][task]['labels']
                version = 0
                epoch = 0
            elif model == 'InceptionNet':
                labels = model_dict[model][task]['labels']
                m_name='inception'

                if project == 'RCC':
                    version = model_dict[model][task]['version']
                    epoch = model_dict[model][task]['epoch']
                else: 
                    version = 0
                    epoch = 0
            probs, target, optimal_threshold = get_data(project=project, model=model, task=task, version=version, epoch=epoch, labels=labels, add_on=0)
            (sorted_auroc, sorted_accuracy, sorted_precision, sorted_recall, sorted_f1), (auroc_ci, accuracy_ci, precision_ci, recall_ci, f1_ci)= bootstrap(probs, target, n=1000)
            # bootstrap(probs, target, n=1000, class_specific=True)
            
            df_list = [sorted_auroc, sorted_accuracy, sorted_precision, sorted_recall, sorted_f1]
            df = pd.DataFrame(df_list, index=['AUROC', 'ACCURACY', 'PRECISION', 'RECALL', 'F1']).T
            out_path = f'{output_path}{project}/'
            Path(out_path).mkdir(exist_ok=True)
            df.to_csv(f'{out_path}/{m_name}_{task}.csv')

            ci_list = [auroc_ci, accuracy_ci, precision_ci, recall_ci, f1_ci]
            ci_list = list(itertools.chain.from_iterable(ci_list))
            complete_ci_list.append(ci_list)
        metric_list = ['AUROC', 'ACCURACY', 'PRECISION', 'RECALL', 'F1']
        columns = []
        for m in metric_list:
            for i in ['LOW', 'HIGH', 'MEAN']:
                columns.append(f'{m}_{i}')
        print(columns)
        ci_df = pd.DataFrame(complete_ci_list, columns=columns, index = model_list)
        ci_df.to_csv(f'{output_path}/{project}/metrics_ci_mean.csv')
        print(ci_df)

if __name__ == '__main__':

    # args = make_parse()
    output_path = f'/homeStor1/ylan/npj_sus_data/figures/sub_figures/'
    
    ##################################
    # Generate AUROC and PRAUC Plots #
    ##################################
    # mode = ['pr', 'roc'] #'roc','pr'
    # # mode = ['roc'] #'roc','pr'
    
    # project = 'DeepGraft'
    # tasks = ['norm_rej_rest'] # 
    # plot_combined(project, mode, tasks, output_path)
    
    # project = 'RCC'
    # tasks = ['big_three']
    # plot_combined(project, mode, tasks, output_path)
    
    
    

    ###########################
    # Generate CI Metric CSVs #
    ###########################

    # models = ['TransMIL', 'CLAM', 'vit', 'Inception']
    
    # project = 'DeepGraft'
    # tasks = ['norm_rej_rest'] # 'norm_rest', 'rej_rest', 
    # output_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/debug/rcc/'

    # plot_combined_cm(tasks, output_path)
    # project = 'RCC'
    # tasks = ['big_three'] # 
    # output_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/bootstrapped_metrics/'
    # get_csv(project, tasks, output_path)

    # project = 'DeepGraft'
    # tasks = ['norm_rej_rest'] # 
    # output_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/bootstrapped_metrics/'
    # get_csv(project, tasks, output_path)
    
    # plot_combined(project, mode, tasks, output_path)
    
    # project = 'DeepGraft'
    # tasks = ['norm_rej_rest']
    # plot_combined(project, mode, tasks, output_path)
    


    