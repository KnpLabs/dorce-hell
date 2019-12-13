from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
import tensorflow_hub as hub

DATA_ROOT='/var/data/'
MODEL_PATH=os.path.join(DATA_ROOT, 'model/pron')

def create_hub_layer():
    return hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", input_shape=[],
                           dtype=tf.string, trainable=True)

def create_model(hub_layer):
    model = tf.keras.Sequential()

    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model