import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, precision_recall_curve, f1_score, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
import xgboost
import math 
from Params import *
from measurements import *
import os
import copy

def GetImagePredictions(test_csv, label, embeddings_folder):
    test_dict = LabelDict(test_csv, label)
    return LoadImagePredictions(embeddings_folder, test_dict, label)
  
def LabelDict(targets, label):
    label_path = os.path.normpath(targets)
    data_df = pd.read_csv(label_path)
    label_dict = data_df.set_index('accession_number')[label].to_dict()

    return label_dict

def GetImageLabel(label_dict, series, label):
    return np.array(label_dict[int(series)])

def LoadImagePredictions(src, test_dict, label):
    
    # Create nparray of embeddings 
    print("Load Image Embeddings")
    X_test_arrays = []    
    y_test_arrays = []
    series_arrays = []
    npfiles = [ f for f in listdir(src) if isfile(join(src,f)) ]

    for f in npfiles:
        if f.endswith('.npy'):
            series = f[:-4]
            if int(series) in test_dict.keys():
                X_test_arrays.append(np.load(src + f))
                y_test_arrays.append(GetImageLabel(test_dict, series, label))
                series_arrays.append(series)


    X_test = np.array(X_test_arrays)
    y_test = np.array(y_test_arrays)
    return X_test, y_test, series_arrays

def ClassifyPredictions(X_test, y_test, series):
    
    print("Classify Image Embeddings ")
    predictions = X_test

    y_test = y_test.reshape(y_test.shape[0],1)
    predictions = predictions.reshape(predictions.shape[0],1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    ci_lower, ci_higher = get_ci_auc(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    print("auc = ",roc_auc)
    print(" 95%  AUC CI bootstrapping : [{:.3f}, {:.3f}]".format(ci_lower, ci_higher))

    alpha = .95

    J = tpr - fpr
    ix = np.argmax(J)
    thresh = thresholds[ix]

    thresh_predictions = copy.deepcopy(predictions)
    print(" Youden index threshold = ", thresh)
    thresh_predictions [predictions > thresh] = 1
    thresh_predictions [predictions <= thresh] = 0
    dice, ppv, sens, acc , npv, spec, tp, fn, fp, tn = analyze_results(thresh_predictions , y_test)
    print("tp={}, fn={}, fp={}, tn={}".format(tp, fn, fp, tn))
    print("dice={:.3f}, ppv={:.3f}, sensitivity={:.3f}, accuracy={:.3f} , npv={:.3f} , specificity={:.3f}"
      .format(dice, ppv, sens, acc , npv, spec))
    correct=[]
    wrong = []
    for idx in (np.where(thresh_predictions == y_test)[0]):
        correct.append(series[idx])
    print("correct classification: ", correct)

    for idx in (np.where(thresh_predictions != y_test)[0]):
        wrong.append(series[idx])
    print("wrong classification: ", wrong)

    return fpr, tpr, roc_auc

def main():
    test_csv = ROOT + TEST_LABELS
    predictions_folder = './results/predictions/'
    label = LABEL_COL
    X_test, y_test, series = GetImagePredictions(test_csv, label, predictions_folder)
    fpr, tpr, roc_auc = ClassifyPredictions(X_test, y_test, series)
    roc_curve = [roc_auc, fpr, tpr]
    roc_auc_np = np.array(roc_auc)
    fpr_np = np.array(fpr)
    tpr_np = np.array(tpr)
    roc_curve_np = np.array(roc_curve)
    if not os.path.exists("./results/auroc"):
        os.makedirs("./results/auroc")
    np.save("./results/auroc/roc_curve.npy", roc_curve_np, allow_pickle=True)

if __name__ == '__main__':
    main()