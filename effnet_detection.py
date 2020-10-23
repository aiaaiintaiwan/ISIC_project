import os, re, time, tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from efficientnet import tfkeras as efnet
from tensorflow.keras.models import load_model
from keras.preprocessing import image

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def build_model(dim=128, ef=0):
    inp = keras.layers.Input(shape=(dim,dim,3))
    base = getattr(efnet, 'EfficientNetB%d' % ef)(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
    x = base(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dense(1024)(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512,activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256,activation='relu')(x)
#     x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256,activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
#     x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128,activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(1)(x)
    x = keras.layers.Activation('sigmoid', dtype='float32')(x)
    model = keras.Model(inputs=inp,outputs=x)
    opt = keras.optimizers.Adam(learning_rate=1e-3)
#     opt = keras.optimizers.SGD(learning_rate=1e-3)

    loss = keras.losses.BinaryCrossentropy(**LOSS_PARAMS)

    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
    return model

if __name__ == "__main__":
    EFF_NET = 0
    IMG_SIZE =384

    LOSS_PARAMS  = dict(label_smoothing=0.09)
    MODEL_WEIGHT_PATH='./effcientB0.h5'

    # model = build_model(dim=IMG_SIZE, ef=EFF_NET)
    # model.load_weights(MODEL_WEIGHT_PATH)

    model = load_model(MODEL_WEIGHT_PATH)
    img_path = './input_data/test.jpg'
    new_image = load_image(img_path, show=False)
    print(new_image.shape)
    test_pred = model.predict(new_image, verbose=1)
    print(test_pred)
