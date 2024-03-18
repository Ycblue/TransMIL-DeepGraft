import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
plt.style.use('tableau-colorblind10')
from matplotlib.colors import ListedColormap
import scienceplots
import pandas as pd
import json
from pprint import pprint
import seaborn as sns
import torch

import torchmetrics
from torchmetrics import PrecisionRecallCurve, ROC
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve, confusion_matrix
from torchmetrics.functional.classification import binary_accuracy, multiclass_accuracy, binary_recall, binary_precision, multiclass_recall, multiclass_precision, binary_f1_score, multiclass_f1_score
from torchmetrics.utilities.compute import _auc_compute_without_check, _auc_compute

from utils.utils import get_roc_curve, get_pr_curve, get_confusion_matrix, get_optimal_operating_point

from scipy.stats import bootstrap as scp_bootstrap
from sklearn.metrics import roc_auc_score

import statistics
import logging

from cycler import cycler

plt.rcParams["font.family"] = "Arial"

renwableRatioCSVPath = '/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/co2_emission/countryRenewableRatio.csv'
re_ratio = pd.read_csv(renwableRatioCSVPath)
re_ratio.sort_values(by=['renewableRatio'], inplace=True)

country_list = list(re_ratio.Country)
germany_ratio = re_ratio.loc[re_ratio.Country == 'Germany']['renewableRatio'].values[0]
ratio_list = [germany_ratio/re_ratio.loc[re_ratio.Country == c]['renewableRatio'].values[0] for c in country_list]

# df.to_csv('/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/co2_emission/model_co2_upper_lower.csv')
co2_range_df = pd.read_csv('/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/co2_emission/model_co2_upper_lower.csv')
mean_model_co2 = co2_range_df['mean_co2']
low_model_co2 = co2_range_df['low_co2']
high_model_co2 = co2_range_df['high_co2']


COLOR_MAP = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]+1, f'{y[i]:.2f}', ha = 'center', fontsize=10)
model_list = ['TransMIL', 'CLAM', 'Inception', 'ViT']
line_styles = ['o-','o--','o:','o-.', 'o-']

paper_outdir = '/homeStor1/ylan/DeepGraft_project/DeepGraft_Draft/figures'


with plt.style.context(['science', 'nature']):
    fig, ax = plt.subplots(figsize=(20,10)) #
    legend_list = []
    # for i, model in enumerate(model_list): #
    # model_list = ['TransMIL', 'CLAM']
    for i, model in enumerate(model_list): #
    # 
        # if model == 'CLAM':
        #     continue
        # model = 'TransMIL'
        color = COLOR_MAP[i]
        
        mean_co2 = [mean_model_co2[i]*r for r in ratio_list]
        low_co2 = [low_model_co2[i]*r for r in ratio_list]
        high_co2 = [high_model_co2[i]*r for r in ratio_list]
        x = country_list

        p, = ax.plot(x, mean_co2, 'o-', color=color, linewidth=3)
        ax.fill_between(x, low_co2, high_co2, color=color, alpha=.15)
        ax.grid(visible=True, which='major', axis='x')
        # ax.set_xticklabels([])
        legend_list.append(p)
            
    # plt.ylim([0.0, 1.0])
    plt.legend(legend_list, model_list, fontsize=30, loc='upper right')
    # plt.xticks([])
    # plt.grid(visible=True, which='major', axis='x')
    plt.gca().xaxis.grid(True)
    plt.ylabel('CO2eq (g)', fontsize= 30)
    # plt.xticks(ha='center', fontsize=30)
    plt.xticks(rotation=90, ha='center', fontsize=30)
    
    
    # plt.xticks(rotation=45, ha='right', fontsize=20)

    plt.yticks(fontsize=30)
    # plt.xticks(ha='right', fontsize=20)
    # plt.title(f'ESPer with lower and upper bound', fontsize=40)
    plt.title(f'CO2eq cost for inference per Country', fontsize=40)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f'{paper_outdir}/co2eq_per_country.png', dpi=400)
    plt.savefig(f'{paper_outdir}/co2eq_per_country.svg', dpi=400)
    plt.show()