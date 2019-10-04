# -*- coding: utf-8 -*-

import keras.backend as K
import tensorflow as tf

def dice_coef_binary(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_binary_loss(y_true, y_pred):
    return 1-dice_coef_binary(y_true, y_pred)

def dice_coef_multiclass(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 3 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_multiclass_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_multiclass(y_true, y_pred)

##MYDICE
def dice_fp_penalty(y_true, y_pred, smooth=1e-7):

    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    fp = K.sum((1.-y_true_f)*y_pred_f, axis=-1)

    return K.mean((2. * intersect / (denom + 100*fp + smooth)))

def dice_fp_penalty_loss(y_true, y_pred):
    return 1 - dice_fp_penalty(y_true, y_pred)

######MYDICE
def dice_fpfn_weighted(y_true, y_pred, smooth=1e-7):

    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
#    tp = K.sum(y_true_f * y_pred_f, axis=-1)
    fp = K.sum((1.-y_true_f)*y_pred_f, axis=-1)
    fn = K.sum(y_true_f*(1.-y_pred_f), axis=-1)

    return K.mean((2. * intersect / (denom + 0.5*(fp+fn) + smooth)))

def dice_fpfn_weighted_loss(y_true, y_pred):
    return 1 - dice_fpfn_weighted(y_true, y_pred)

###############
def mIU_fp_penalty(y_true, y_pred, smooth=1e-7):
    
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    
    tp = K.sum(y_true_f * y_pred_f, axis=-1)
    fp = K.sum((1.-y_true_f)*y_pred_f, axis=-1)
    fn = K.sum(y_true_f*(1.-y_pred_f), axis=-1)
    
    return K.mean((tp / (tp + fp + fn + smooth)))


def mIU_fp_penalty_loss(y_true, y_pred):    
    return 1 - mIU_fp_penalty(y_true, y_pred)


def focal_loss(y_true, y_pred, gamma=2, alpha=0.75):
    eps = 1e-12
    y_pred=K.clip(y_pred,eps,1.-eps) #improve the stability of the focal loss and see issues 1 for more information
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def iou_loss_score(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    print(intersection)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    print(union)
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def dice_iou_loss(y_true, y_pred):
#    return (iou_loss_score(y_true, y_pred) + dice_coef_multiclass_loss(y_true, y_pred))/2.
    return 1 - (mIU_fp_penalty(y_true, y_pred) + dice_coef_multiclass(y_true, y_pred))/2.

def dice_iou_conditional_loss(y_true, y_pred):
    iou = mIU_fp_penalty(y_true, y_pred)
    dice = dice_coef_multiclass(y_true, y_pred)
    print(iou)
#    b =  K.eval(dice)
#    print(a)
    return 1 - dice
#    if a >= b:
#        return 1 - dice
#    else:
#        return 1 - iou
    
def dice_iou_weighted_loss(y_true, y_pred):
    iou = mIU_fp_penalty(y_true, y_pred)
    dice = dice_coef_multiclass(y_true, y_pred)
    if iou >= dice:
        return 
    else:
        return 1 - (dice/iou*iou + iou/dice*dice)/2.

#np.median(np.asarray([iou, dice])/sum(np.asarray([iou, dice])))/(np.asarray([iou, dice])/sum(np.asarray([iou, dice])))