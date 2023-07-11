# TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification [NeurIPS 2021]
## Abstract
Multiple instance learning (MIL) is a powerful tool to solve the weakly supervised classification in whole slide image (WSI) based pathology diagnosis. However, the current MIL methods are usually based on independent and identical distribution hypothesis, thus neglect the correlation among different instances. To address this problem, we proposed a new framework, called correlated MIL, and provided a proof for convergence. Based on this framework, we devised a Transformer based MIL (TransMIL), which explored both morphological and spatial information. The proposed TransMIL can effectively deal with unbalanced/balanced and binary/multiple classification with great visualization and interpretability. We conducted various experiments for three different computational pathology problems and achieved better performance and faster convergence compared with state-of-the-art methods. The test AUC for the binary tumor classification can be up to 93.09% over CAMELYON16 dataset. And the AUC over the cancer subtypes classification can be up to 96.03% and 98.82% over TCGA-NSCLC dataset and TCGA-RCC dataset, respectively.
### Train
```python
python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```
### Test
```python
python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```


### Changes Made: 

### Baseline: 

lr = 0.0002
wd = 0.01

| task        | main | backbone | train_auc | val_auc | epochs | version |  
|---|---|---|---|---|
| tcmr_viral | TransMIL | resnet50 |  0.997 | 0.871 | 200 | 4 |
|            |          | resnet18 |  0.999 | 0.687 | 200 | 0 |
|            |          | efficientnet | 0.99 | 0.76 | 200 | 107 |
|            | DTFD     | resnet50 | 0.989 | 0.621 | 200 | 44 |
|            | AttMIL   | simple | 0.513 | 0.518 | 200 | 50 |


159	28639			0.9222221970558167	0.19437336921691895	0.5906432867050171	0.56540447473526	0.7159091234207153	0.8709122538566589	0.30908203125

### Ablation

# Important things: 

    * 

# Instructions

* Train model
* Validate model to generate topk patient lists for each patient
* Test model
* visualize with visualize_classic.py or visualize_mil.py