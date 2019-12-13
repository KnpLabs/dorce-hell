from __future__ import absolute_import, division, print_function, unicode_literals
import json
import os
import sys

import numpy as np
import tensorflow_datasets as tfds
import keras

import model as mdl

model = mdl.create_model(mdl.create_hub_layer())

model.load_weights(mdl.MODEL_PATH)

prediction = model.predict(np.array(keras.preprocessing.text.text_to_word_sequence(sys.argv[1], split='.')))

print(json.dumps({"score": float(prediction[0][0])}))
