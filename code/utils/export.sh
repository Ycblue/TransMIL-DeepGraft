#!/bin/sh

$model='TransMIL'
$task='norm_rest'

python export_metrics.py --model $model --task $task --target_label 0 
python export_metrics.py --model $model --task $task --target_label 1 

$task='rej_rest'

python export_metrics.py --model $model --task $task --target_label 0 
python export_metrics.py --model $model --task $task --target_label 1 

$task='norm_rej_rest'

python export_metrics.py --model $model --task $task --target_label 0 
python export_metrics.py --model $model --task $task --target_label 1 
python export_metrics.py --model $model --task $task --target_label 2 