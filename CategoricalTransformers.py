import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from scipy.spatial.distance import cdist
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, Flatten
from keras.optimizers import Adam

# from tensorflow.python.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

import ktrain
from ktrain import text
from transformers import BertModel, BertConfig


tf.compat.v1.disable_eager_execution()

# categories:
#   0 - correct
#   1 - sv agreement
#   2 - np error
#   3 - tense
categories = ['0', '1', '2', '3']

x_train_text = []
y_train_targ = []
x_test_text = []
y_test_targ = []

with open("combinetrainset.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        x_train_text.append(row[0])
        y_train_targ.append(row[3])

with open("combinetestset.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        x_test_text.append(row[0])
        y_test_targ.append(row[3])

MODEL_NAME = 'distilbert-base-uncased'

t = text.Transformer(MODEL_NAME, maxlen=500, class_names=categories)

trn = t.preprocess_train(x_train_text, y_train_targ)
val = t.preprocess_test(x_test_text, y_test_targ)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(5e-5, 4)
learner.validate(class_names=t.get_classes())

# learner.lr_find(show_plot=True, max_epochs=1)

