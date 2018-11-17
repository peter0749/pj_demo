import os
import gc
from .config import *
from .losses import *
from .metrics import *
import numpy as np

### U-Net:
def get_U_Net_model(gpus=1, load_weights=None, verbose=False):
    from keras.models import Model, load_model
    from keras.layers import Input, Add, Activation, Dropout
    from keras.layers.core import Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from .weightnorm import AdamWithWeightnorm
    from keras.utils import multi_gpu_model
    from keras import backend as K
    import tensorflow as tf

    IMG_WIDTH = U_NET_DIM
    IMG_HEIGHT= U_NET_DIM
    IMG_CHANNELS= 3

    def conv(f, k=3, act='elu'):
        return Conv2D(f, (k, k), activation=act, kernel_initializer='he_normal', padding='same')
    def _incept_conv(inputs, f, chs=[0.15, 0.5, 0.25, 0.1]):
        fs = [] # determine channel number
        for k in chs:
            t = max(int(k*f), 1) # at least 1 channel
            fs.append(t)

        fs[1] += f-np.sum(fs) # reminding channels allocate to 3x3 conv

        c1x1 = conv(fs[0], 1, act='linear') (inputs)
        c3x3 = conv(max(1, fs[1]//2), 1, act='elu') (inputs)
        c5x5 = conv(max(1, fs[2]//3), 1, act='elu') (inputs)
        cpool= MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same') (inputs)

        c3x3 = conv(fs[1], 3, act='linear') (c3x3)
        c5x5 = conv(fs[2], 5, act='linear') (c5x5)
        cpool= conv(fs[3], 1, act='linear') (cpool)

        output = concatenate([c1x1, c3x3, c5x5, cpool], axis=-1)
        return output

    def _res_conv(inputs, f, k=3): # very simple residual module
        channels = int(inputs.shape[-1])

        cs = _incept_conv(inputs, f)

        if f!=channels:
            t1 = conv(f, 1, 'linear') (inputs) # identity mapping
        else:
            t1 = inputs

        out = Add()([t1, cs]) # t1 + c2
        out = Activation('elu') (out)
        out = Dropout(0.2) (out)
        return out
    def pool():
        return MaxPooling2D((2, 2))
    def up(inputs):
        upsampled = Conv2DTranspose(int(inputs.shape[-1]), (2, 2), strides=(2, 2), padding='same') (inputs)
        return upsampled

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = _res_conv(inputs, 32, 3)
    c1 = _res_conv(c1, 32, 3)
    p1 = pool() (c1)

    c2 = _res_conv(p1, 64, 3)
    c2 = _res_conv(c2, 64, 3)
    p2 = pool() (c2)

    c3 = _res_conv(p2, 128, 3)
    c3 = _res_conv(c3, 128, 3)
    p3 = pool() (c3)

    c4 = _res_conv(p3, 256, 3)
    c4 = _res_conv(c4, 256, 3)
    p4 = pool() (c4)

    c5 = _res_conv(p4, 512, 3)
    c5 = _res_conv(c5, 512, 3)
    p5 = pool() (c5)

    c6 = _res_conv(p5, 512, 3)
    c6 = _res_conv(c6, 512, 3)

    u7 = up (c6)
    c7 = concatenate([u7, c5])
    c7 = _res_conv(c7, 512, 3)
    c7 = _res_conv(c7, 512, 3)

    u8 = up (c7)
    c8 = concatenate([u8, c4])
    c8 = _res_conv(c8, 256, 3)
    c8 = _res_conv(c8, 256, 3)

    u9 = up (c8)
    c9 = concatenate([u9, c3])
    c9 = _res_conv(c9, 128, 3)
    c9 = _res_conv(c9, 128, 3)

    u10 = up (c9)
    c10 = concatenate([u10, c2])
    c10 = _res_conv(c10, 64, 3)
    c10 = _res_conv(c10, 64, 3)

    u11 = up (c10)
    c11 = concatenate([u11, c1])
    c11 = _res_conv(c11, 32, 3)
    c11 = _res_conv(c11, 32, 3)

    netout = Conv2D(3, (1, 1), activation='sigmoid', name='nuclei') (c11)
    optimizer = AdamWithWeightnorm(**U_NET_OPT_ARGS)
    with tf.device('/cpu:0'): ## prevent OOM error
        cpu_model = Model(inputs=[inputs], outputs=[netout])
        cpu_model.compile(loss=custom_loss, metrics=[mean_iou, mean_iou_marker], optimizer=optimizer)
        if load_weights is not None and os.path.exists(load_weights):
            cpu_model.load_weights(load_weights)
            if verbose:
                print('Loaded weights')
    if gpus>=2:
        gpu_model = multi_gpu_model(cpu_model, gpus=gpus)
        gpu_model.compile(loss=custom_loss, metrics=[mean_iou, mean_iou_marker], optimizer=optimizer)
        return gpu_model, cpu_model
    return cpu_model, cpu_model
### end U-Net model
