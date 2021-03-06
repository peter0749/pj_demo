from __future__ import unicode_literals
import os
import copy
import numpy as np
from django.db import models
import cv2
from .utils import rle_encoding, get_label, lb, label_to_rles
from skimage.segmentation import mark_boundaries
from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model

def dummy_f(a,b):
    return 0

keras.backend.clear_session()
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfsession = tf.Session(config=tfconfig)
keras.backend.set_session(tfsession)
unet_model = load_model('ensemble.h5', custom_objects={'mean_iou': dummy_f, 'custom_loss': dummy_f, 'mean_iou_marker': dummy_f} )
unet_model.compile(loss='mean_squared_error', optimizer='sgd')
unet_model.summary()
graph = tf.get_default_graph()
input_h, input_w = unet_model.input_shape[1:3]

class IMG(models.Model):
    img = models.ImageField()
    def __init__(self, *args, **kwargs):
        super(IMG, self).__init__(*args, **kwargs)
        self.original_filename = os.path.split(self.img.path)[-1]
        self.download_filename = os.path.splitext(self.original_filename)[0] + '_wrapped.png'

    def save(self):
        super(IMG, self).save() # if has same filename. change to different filename
        self.path = self.img.path
        t = os.path.splitext(self.img.path)
        self.wrap_path = t[0] + '_marked.png'
        self.url_path = self.img.url
        t = os.path.splitext(self.img.url)
        self.wrap_url_path = t[0] + '_marked.png'
        raw_image = Image.open(self.img)
        raw_image = raw_image.convert('RGB')
        raw_image = np.array(raw_image, dtype=np.uint8)
        img_r = cv2.resize(raw_image, (input_w, input_h), interpolation=cv2.INTER_AREA).astype(np.float32)[np.newaxis,...]
        h, w = raw_image.shape[:2]
        with graph.as_default():
            preds  = unet_model.predict(img_r, batch_size=1)
            nuclei = cv2.resize(preds[0,...,0], (w, h))
            marker = cv2.resize(preds[0,...,1], (w, h))
            dt     = cv2.resize(preds[0,...,2], (w, h))
        del preds
        label = get_label(nuclei, marker, dt)
        marked_img = np.round(np.clip(mark_boundaries(raw_image.astype(np.float32)/255.0, label, mode='outer') * 255, 0, 255)).astype(np.uint8)
        result = Image.fromarray(marked_img)
        result.save(self.wrap_path)
        del label, nuclei, marker
