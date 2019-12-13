from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import keras

import model as mdl

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

DATASET_SIZE = 500

lines_dataset = tf.data.TextLineDataset(os.path.join(mdl.DATA_ROOT, 'pron.txt'))
porn_labeled_data_set = lines_dataset.map(lambda example: (example, True))

lines_dataset = tf.data.TextLineDataset(os.path.join(mdl.DATA_ROOT, 'not-pron.txt'))
notporn_labeled_data_set = lines_dataset.map(lambda example: (example, False))

dataset = porn_labeled_data_set.concatenate(notporn_labeled_data_set)
dataset = dataset.shuffle(DATASET_SIZE, reshuffle_each_iteration=False)

train_data = dataset.shard(3, 0)
validation_data = dataset.shard(3, 1)
test_data = dataset.shard(3, 2)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

hub_layer = mdl.create_hub_layer()
hub_layer(train_examples_batch[:3])

model = mdl.create_model(hub_layer)

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=60,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

model.save_weights(mdl.MODEL_PATH)

print('Model saved!')