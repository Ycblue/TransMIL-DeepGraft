# Paper Outline

## Abstract

Sustainability is important. Therefore a sustainability-minded approach to benchmarking DL models should be employed.
We benchmark 4 models on two tasks, DeepGraft and RCC classification to show how.
Additionally we introduce a new metric, combining accuracy and CO2eq which should help better understand the impact of each model.

## Introduction

Sustainability is important. Therefore a sustainability-minded approach to benchmarking DL models should be employed.
We benchmark 4 models on two tasks, DeepGraft and RCC classification to show how.
Additionally we introduce a new metric, combining accuracy and CO2eq which should help better understand the impact of each model.

## Methods

Multiple Instance Learning

4 Models: TransMIL, CLAM, ViT, Inception

Task 1: Kidney Transplant Disease Classification 3 class

Task 2: RCC Classification for ccRCC, chRCC, papRCC

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
