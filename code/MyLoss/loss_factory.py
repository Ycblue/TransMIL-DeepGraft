__author__ = 'shaozc'

import torch
import torch.nn as nn

from .boundary_loss import BDLoss, SoftDiceLoss, DC_and_BD_loss, HDDTBinaryLoss,\
     DC_and_HDBinary_loss, DistBinaryDiceLoss
from .dice_loss import GDiceLoss, GDiceLossV2, SSLoss, SoftDiceLoss,\
     IoULoss, TverskyLoss, FocalTversky_loss, AsymLoss, DC_and_CE_loss,\
         PenaltyGDiceLoss, DC_and_topk_loss, ExpLog_loss
from .focal_loss import FocalLoss
from .focal_loss_ori import FocalLoss_Ori
from .hausdorff import HausdorffDTLoss, HausdorffERLoss
from .lovasz_loss import LovaszSoftmax
from .ND_Crossentropy import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss,\
     WeightedCrossEntropyLossV2, DisPenalizedCE
from .poly_loss import PolyLoss

from pytorch_toolbelt import losses as L

def create_loss(args, n_classes, w1=1.0, w2=0.5):
    conf_loss = args.base_loss
    # n_classes = args.model.n_classes
    # if args.loss_weight: 
    #     weight = torch.tensor(args.loss_weight)
    # else: weight = None
    ### MulticlassJaccardLoss(classes=np.arange(11)
    # mode = args.base_loss #BINARY_MODE \MULTICLASS_MODE \MULTILABEL_MODE 
    loss = None
    # print(conf_loss)
    if hasattr(nn, conf_loss): 
        loss = getattr(nn, conf_loss)()
        # loss = getattr(nn, conf_loss)(label_smoothing=0.1) 
    #binary loss
    elif conf_loss == "focal":
        loss = FocalLoss_Ori(n_classes)
    elif conf_loss == "jaccard":
        loss = L.BinaryJaccardLoss()
    elif conf_loss == "jaccard_log":
        loss = L.BinaryJaccardLoss()
    elif conf_loss == "dice":
        loss = L.BinaryDiceLoss()
    elif conf_loss == "dice_log":
        loss = L.BinaryDiceLogLoss()
    elif conf_loss == "bce+lovasz":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryLovaszLoss(), w1, w2)
    elif conf_loss == "lovasz":
        loss = L.BinaryLovaszLoss()
    elif conf_loss == "bce+jaccard":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryJaccardLoss(), w1, w2)
    elif conf_loss == "bce+log_jaccard":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryJaccardLogLoss(), w1, w2)
    elif conf_loss == "bce+log_dice":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryDiceLogLoss(), w1, w2)
    elif conf_loss == "reduced_focal":
        loss = L.BinaryFocalLoss(reduced=True)
    elif conf_loss == "polyloss":
        loss = PolyLoss(softmax=False)
    else:
        assert False and "Invalid loss"
        raise ValueError
    return loss

import argparse
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-loss', default='CrossEntropyLoss',type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = make_parse()
    myloss = create_loss(args)
    print(myloss)
    data = torch.randn(2, 3)
    label = torch.empty(2, dtype=torch.long).random_(3)
    loss = myloss(data, label)