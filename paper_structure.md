# Paper Outline

## Abstract

## Introduction

Why do we do this

## Methods

    - Fig: Study design
        * Choosing the data
        * Model workflow 
    
    Groud truth
    DL Analyses

    - 

## Dataset

    - Fig: cohorts, data selection, Train/Val/Test Split
    - Fig: Preprocessing, Annotations

## Results

Compare models on Normal vs Rest and then choose best Model.

    - Fig: Metrics on Testset for Normal vs Rejection vs Rest:


| Model        | Accuracy | Precision | Recall | AUROC |
| ------------ | -------- | --------- | ------ | ----- |
|              |          |           |        |       |
| ViT          |          |           |        |       |
| CLAM         |          |           |        |       |
| AttentionMIL |          |           |        |       |
| TransMIL     |          |           |        |       |

    - Fig: Metrics on Testset for Normal vs Rest + Rejections vs Rest:

| Model        | Accuracy | Precision | Recall | AUROC |
| ------------ | -------- | --------- | ------ | ----- |
|              |          |           |        |       |
| ViT          |          |           |        |       |
| CLAM         |          |           |        |       |
| AttentionMIL |          |           |        |       |
| TransMIL     |          |           |        |       |

 - Two step Models are still better
    - Fig: AUROC Curves (Best Model, Rest in Appendix)
    - Fig: Attention Maps (Best Model, Rest in Appendix)

- Single Model: 
    AUROC Curves for each class
    Prediction Maps on slide + Scale
    Predictive tiles with Original and GradCam + Scale

- Two step Model: 
    AUROC Curves for each class
    Prediction Maps on slide + Scale
    Predictive tiles with Original and GradCam + Scale

## Discussion

## Appendix

### Sustainability Study

### Fine Tuning on Test Set
