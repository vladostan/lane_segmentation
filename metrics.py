# coding: utf-8
import numpy as np
from keras.utils import to_categorical

def tpfpfn(pred_labels, true_labels):

    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    return TP, FP, FN, TN

def Accuracy(TP, FP, FN, TN):
    
    if TP == 0 and TN == 0:
        return 0
    
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    
    return Accuracy

def Precision(TP, FP):
    
    if TP == 0:
        return 0
    
    Precision = TP/(TP+FP)
    
    return Precision

def Recall(TP, FN):

    if TP == 0:
        return 0
    
    Recall = TP/(TP+FN)
    
    return Recall

def IU(TP, FP, FN):
    
    if TP == 0:
        return 0
    
    IU = TP/(TP+FP+FN)
    
    return IU

def F1(TP, FP, FN):
    
    if TP == 0:
        return 0
    
    F1 = 2*TP/(2*TP + FP + FN)
    
    return F1

##########################
def TNR(TN, FP):
    
    if TN == 0:
        return 0
    
    TNR = TN/(TN + FP)
    
    return TNR

def NPV(TN, FN):
    
    if TN == 0:
        return 0
    
    NPV = TN/(TN + FN)
    
    return NPV

def FPR(FP, TN):
    
    if FP == 0:
        return 0
    
    FPR = FP/(FP + TN)
    
    return FPR

def FDR(FP, TP):
    
    if FP == 0:
        return 0
    
    FDR = FP/(FP + TP)
    
    return FDR

def FNR(FN, TP):
    
    if FN == 0:
        return 0
    
    FNR = FN/(FN + TP)
    
    return FNR

def BACC(TP, FP, FN, TN):
        
    BACC =  (Recall(TP, FN) + TNR(TN, FP))/2
    
    return BACC

###########################
def mAccuracy(y_pred, y_true):
    
    mAccuracy = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, FP, FN, TN = tpfpfn(pred_labels, true_labels)
        mAccuracy += Accuracy(TP, FP, FN, TN)/2
        
    return mAccuracy

def mPrecision(y_pred, y_true):
    
    mPrecision = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, FP, _, _ = tpfpfn(pred_labels, true_labels)
        mPrecision += Precision(TP, FP)/2
        
    return mPrecision

def mRecall(y_pred, y_true):
    
    mRecall = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, _, FN, _ = tpfpfn(pred_labels, true_labels)
        mRecall += Recall(TP, FN)/2
        
    return mRecall

def mIU(y_pred, y_true):
    
    mIU = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, FP, FN, _ = tpfpfn(pred_labels, true_labels)
        mIU += IU(TP, FP, FN)/2
        
    return mIU

def mF1(y_pred, y_true):
    
    mF1 = 0

    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, FP, FN, _ = tpfpfn(pred_labels, true_labels)
        mF1 += F1(TP, FP, FN)/2

    return mF1

def mTNR(y_pred, y_true):
    
    mTNR = 0

    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        _, FP, _, TN = tpfpfn(pred_labels, true_labels)
        mTNR += TNR(TN, FP)/2

    return mTNR

def mNPV(y_pred, y_true):
    
    mNPV = 0

    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        _, _, FN, TN = tpfpfn(pred_labels, true_labels)
        mNPV += NPV(TN, FN)/2

    return mNPV

def mFPR(y_pred, y_true):
    
    mFPR = 0

    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        _, FP, _, TN = tpfpfn(pred_labels, true_labels)
        mFPR += FPR(TN, FP)/2

    return mFPR

def mFDR(y_pred, y_true):
    
    mFDR = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, FP, _, _ = tpfpfn(pred_labels, true_labels)
        mFDR += FDR(TP, FP)/2
        
    return mFDR

def mFNR(y_pred, y_true):
    
    mFNR = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, _, FN, _ = tpfpfn(pred_labels, true_labels)
        mFNR += FNR(TP, FN)/2
        
    return mFNR

def mBACC(y_pred, y_true):
    
    mBACC = 0
    
    # Calculate per class, ignoring background
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        TP, FP, FN, TN = tpfpfn(pred_labels, true_labels)
        mBACC += BACC(TP, FP, FN, TN)/2
        
    return mBACC