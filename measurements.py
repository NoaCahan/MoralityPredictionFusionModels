import cv2
from os import listdir
from os.path import isfile, join
import re
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy.stats
from scipy import stats
from Params import NUM_CLASSES
from sklearn.metrics import auc, roc_auc_score, roc_curve

def plot_feat_importance(tabnet_model):
    """Plot TabNet feature importance scores.
    Args:
        tabnet_model: The trained TabNet model.
    Returns:
        Plot of all features arranged by magnitude.
    """
    feat_importances = tabnet_model.feature_importances_
    indices = np.argsort(feat_importances)
    end = len(feat_importances)
       
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(len(feat_importances[(end-30):end])), feat_importances[indices[(end-30):end]],
           color="r", align="center")

    print([features[idx] for idx in indices[(end-30):end]])
    print(indices[(end-30):end])
    print(feat_importances[indices[(end-30):end]])
    
    plt.yticks(range(len(feat_importances[(end-30):end])), [features[idx] for idx in indices[(end-30):end]])
    plt.ylim([-1, len(feat_importances[(end-30):end])])
    plt.show(block = True)
	
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    for i in range(NUM_CLASSES):
        AUROCs.append(roc_auc_score(gt[:, i], pred[:, i]))
    return AUROCs

def analyze_results(X,y):

    patients = len(y)
    tn, fp, fn, tp = confusion_matrix(y, X).ravel()

    dice = calc_dice(tp, fn, fp)
    sens = calc_sensitivity(tp, fn)
    acc = calc_acc(tp, tn, patients)

    if tp == 0 and fp == 0 :
        ppv = -1
    else:
        ppv = calc_positive_predictive_value(tp, fp)
    npv = calc_negative_predictive_value(fn, tn)
    spec = calc_specificity(fp, tn)
    return dice, ppv, sens, acc , npv, spec, tp, fn, fp, tn

def clac_youden_index(y, X):

    tn, fp, fn, tp = confusion_matrix(y, X).ravel()
    sens = calc_sensitivity(tp, fn)
    spec = calc_specificity(fp, tn)
    youden = sens/100 + spec/100 - 1
    return youden
	
def calc_auc_ci(X,y):

    # Calculate the Confidance Interval:
    fpr, tpr, thresholds = roc_curve(y, X)
    ci_lower, ci_higher = get_ci_auc(y, X)
    roc_auc = auc(fpr, tpr)
    print("auc = ",roc_auc)
    print(" 95%  AUC CI bootstrapping : [{:.3f}, {:.3f}]".format(ci_lower, ci_higher))

    alpha = .95
    roc_auc, auc_cov = delong_roc_variance(y, X)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=roc_auc,
        scale=auc_std)

    ci[ci > 1] = 1

    print('auc cov:', auc_cov)
    print('95% AUC CI DeLong:', ci)

    J = tpr - fpr
    ix = np.argmax(J)

    thresh = thresholds[ix]
    print("Youden Index " ,thresh)
    return ci_lower, ci_higher

def get_ci_auc( y_true, y_pred ): 

    from scipy.stats import sem
    from sklearn.metrics import roc_auc_score 
   
    np.random.seed(1234)
    rng=np.random.RandomState(1234)

    n_bootstraps = 1000   
    bootstrapped_scores = []   
   
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)   
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% c.i.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower,confidence_upper

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight = None):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (~ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

def calc_dice(tp, fn, fp):
    return 200.0 * tp / (2 * tp + fn + fp)

#AKA ppv or precision
def calc_positive_predictive_value(tp, fp):
    return 100.0 * (tp / (tp + fp))

# AKA TPR or Recall
def calc_sensitivity(tp, fn):
    return 100.0 * (tp / (tp + fn))

def calc_acc(tp, tn, patients):
    return 100* ((tp + tn) / patients)

def calc_negative_predictive_value(fn, tn):
    return 100.0 * (tn / (tn + fn))

def calc_specificity(fp, tn):
    return 100.0 * (tn / (tn + fp))

if __name__ == "__main__":
    analyze_results(debug=True)