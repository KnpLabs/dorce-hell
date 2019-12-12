from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

import os

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

DATA_ROOT = '/var/data/'
DATASET_SIZE = 500

lines_dataset = tf.data.TextLineDataset(os.path.join(DATA_ROOT, 'pron.txt'))
porn_labeled_data_set = lines_dataset.map(lambda example: (example, True))

lines_dataset = tf.data.TextLineDataset(os.path.join(DATA_ROOT, 'not-pron.txt'))
notporn_labeled_data_set = lines_dataset.map(lambda example: (example, False))

dataset = porn_labeled_data_set.concatenate(notporn_labeled_data_set)
dataset = dataset.shuffle(DATASET_SIZE, reshuffle_each_iteration=False)

train_data = dataset.shard(3, 0)
validation_data = dataset.shard(3, 1)
test_data = dataset.shard(3, 2)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
