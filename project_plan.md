#   Benchmarking weakly supervised deep learning models for transplant pathology classification

With this project, we aim to esatablish a benchmark for weakly supervised deep learning models for transplant pathology classification, especially for multiple instance learning approaches. 


## Cohorts:

#### Original Lancet Set:

    * Training:
        * AMS: 1130 Biopsies (3390 WSI)
        * Utrecht: 717 Biopsies (2151WSI)
    * Testing:
        * Aachen: 101 Biopsies (303 WSI)


#### Extended:

* Training:
  * AMS + Utrecht + Leuven
* Testing:
  * Aachen_extended:

## Models:

    For our Benchmark, we chose the following models: 

    - AttentionMIL
    - Resnet18/50
    - ViT
    - CLAM
    - TransMIL
    - Monai MIL (optional)

    Resnet18 and Resnet50 are basic CNNs that can be applied for a variety of tasks. Although domain or task specific architectures mostly outperform them, they remain a good baseline for comparison. 

    The vision transformer is the first transformer based model that was adapted to computer vision tasks. Benchmarking on ViT can provide more insight on the performance of generic transformer based models on multiple instance learning. 

    The AttentionMIL was the first simple, yet relatively successful deep MIL model and should be used as a baseline for benchmarking MIL methods. 

    CLAM is a recent model proposed by Mahmood lab which was explicitely trained for histopathological whole slide images and should be used as a baseline for benchmarking MIL methods in histopathology. 

    TransMIL is another model proposed by Shao et al, which achieved SOTA on histopathological WSI classification tasks using MIL. It was benchmarked on TCGA and compared to CLAM and AttMIL. It utilizes the self-attention module from transformer models.

    Monai MIL (not official name) is a MIL architecture proposed by Myronenk et al (Nvidia). It applies the self-attention mechanism as well. It is included because it shows promising results and it's included in MONAI. 

## Tasks:

    The Original tasks mimic the ones published in the original DeepGraft Lancet paper. 
    Before we go for more challenging tasks (future tasks), we want to establish that our models outperform the simpler approach from the previous paper and that going for MIL in this setting is indeed profitable. 

    All available classes: 
        * Normal
        * TCMR
        * ABMR
        * Mixed
        * Viral
        * Other

#### Original:

    The explicit classes are simplified/grouped together such as this: 
    Diseased = all classes other than Normal 
    Rejection = TCMR, ABMR, Mixed 

    - (1) Normal vs Diseased (all other classes)
    - (2) Rejection vs (Viral + Others)
    - (3) Normal vs Rejection vs (Viral + Others)

#### Future:

    After validating Original tasks, the next step is to challenge the models by attempting more complicated tasks. 
    These experiments may vary depending on the results from previous experiments

    - (4) Normal vs TCMR vs Mixed vs ABMR vs Viral vs Others
    - (5) TCMR vs Mixed vs ABMR

## Plan:

    1. Train models for current tasks on AMS+Utrecht -> Validate on Aachen

    2. Visualization, AUC Curves

    3. Train best model on extended training set (AMS+Utrecht+Leuven) (Tasks 1,2,3) -> Validate on Aachen_extended
        - Investigate if a larger training cohort increases performance
    4. Train best model on extended dataset on future tasks (Task 4, 5)


    Notes: 
        * Resnet18, ViT and CLAM are all trained on HIA (Training Framework from Kather / Narmin)
    

## Status: 

        - Resnet18: Trained on all tasks via HIA  
        - Vit: Trained on all tasks via HIA 
        - CLAM: Trained on (1) via HIA 
        - TransMIL: Trained, but overfitting
            - Check if the problems are not on model side by evaluating on RCC data. 
            - (mixing in 10 slides from Aachen increases auc performance from 0.7 to 0.89)
        - AttentionMIL: WIP
        - Monai MIL: WIP
