import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from itertools import repeat
import numpy as np

def classification_loss(output, target, pos_weight=None):
    BCE = F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
    return BCE

def multilabel_classification_loss(output, target, pos_weight=None):
    BCE = F.cross_entropy(output, target, pos_weight)
    return BCE

ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky