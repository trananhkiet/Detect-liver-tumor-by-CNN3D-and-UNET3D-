from keras import backend as K
import numpy  as np


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def IOU(y_true, y_pred, smooth = 1e-8):
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = y_true_f * y_pred_f
    union = y_true_f + y_pred_f - intersection
    return (K.sum(intersection) + smooth) / (K.sum(union) + smooth)
    
def VOE(y_true, y_pred, smooth = 1e-8):
    return 1. - IOU(y_true, y_pred, smooth)
