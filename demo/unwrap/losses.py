import numpy as np
import tensorflow as tf
import keras.backend as K

def custom_loss(y_true_, y_pred_):

    def dice_coef(y_true, y_pred, intersection): # (batch_size, h, w)
        return (2.*K.sum(intersection) + 1) / (K.sum(y_true) + K.sum(y_pred) + 1) # scaler

    obj_scale = 3
    noobj_scale = 0.1

    y_true_hm = y_true_[...,0] # (batch_size, h, w)
    y_true_mk = y_true_[...,1] # (batch_size, h, w)
    y_true_dt = y_true_[...,2] # (batch_size, h, w)

    y_pred_hm = y_pred_[...,0] # (batch_size, h, w)
    y_pred_mk = y_pred_[...,1] # (batch_size, h, w)
    y_pred_dt = y_pred_[...,2] # (batch_size, h, w)

    true_mask_scale = y_true_hm * (obj_scale-noobj_scale) + noobj_scale
    s_dt_regression_loss = K.sum(K.square(y_true_dt - y_pred_dt) * true_mask_scale, axis=None) / (K.sum(true_mask_scale, axis=None) + K.epsilon()) # scaler

    intersection_scale =  (1 - y_true_dt) * y_true_hm + 1

    mk_intersection = y_true_mk * y_pred_mk
    hm_intersection = y_true_hm * y_pred_hm

    s_marker_dice_coef = dice_coef(y_true_mk * intersection_scale, y_pred_mk * intersection_scale, mk_intersection * intersection_scale) # scaler
    s_heatmap_dice_coef= dice_coef(y_true_hm * intersection_scale, y_pred_hm * intersection_scale, hm_intersection * intersection_scale) # scaler
    s_marker_entropy   = K.sum(K.binary_crossentropy(y_true_mk, y_pred_mk) * intersection_scale, axis=None) / (K.sum(intersection_scale, axis=None) + K.epsilon()) # scaler
    s_heatmap_entropy  = K.sum(K.binary_crossentropy(y_true_hm, y_pred_hm) * intersection_scale, axis=None) / (K.sum(intersection_scale, axis=None) + K.epsilon()) # scaler
    s_marker_loss = .5 * s_marker_entropy - s_marker_dice_coef + 1. # scaler [0, inf)
    s_heatmap_loss= .5 * s_heatmap_entropy - s_heatmap_dice_coef + 1. # scaler [0, inf)

    loss = s_marker_loss + s_heatmap_loss + s_dt_regression_loss # scaler [0, inf)
    loss = tf.Print(loss, [s_marker_dice_coef], message='\nMarker DC:\t', summarize=10)
    loss = tf.Print(loss, [s_marker_entropy], message='\nMarker ET:\t', summarize=10)
    loss = tf.Print(loss, [s_heatmap_dice_coef], message='\nHeatmap DC:\t', summarize=10)
    loss = tf.Print(loss, [s_heatmap_entropy], message='\nHeatmap ET:\t', summarize=10)
    loss = tf.Print(loss, [s_dt_regression_loss], message='\nDT MSE:\t', summarize=10)

    return loss # scaler [0, inf)
