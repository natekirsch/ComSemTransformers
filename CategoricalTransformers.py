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
categories = [0, 1, 2, 3]

x_train_text = []
y_train_targ = []
x_test_text = []
y_test_targ = []

with open("combinetrainset.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        x_train_text.append(row[0])
        y_train_targ.append(float(row[3]))

with open("combinetestset.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        x_test_text.append(row[0])
        y_test_targ.append(float(row[3]))

MODEL_NAME = 'distilbert-base-uncased'

t = text.Transformer(MODEL_NAME, maxlen=500, classes=categories)

trn = t.preprocess_train(x_train_text, y_train_targ)
val = t.preprocess_test(x_test_text, y_test_targ)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
# learner.fit_onecycle(5e-5, 4)

learner.lr_find(show_plot=True, max_epochs=3)

# one_hot_labels = tf.keras.utils.to_categorical(
#     y_train, num_classes=4
# )  # remember to change num classes to reflect number of error types vhl
# two_hot_labels = tf.keras.utils.to_categorical(
#     y_test, num_classes=4
# )  # num classes is number of errors
# data_text = x_train_text + x_test_text

# num_words = 10000
# tokenizer = Tokenizer(num_words=num_words)

# tokenizer.fit_on_texts(data_text)

# x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
# x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

# num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
# num_tokens = np.array(num_tokens)

# max_tokens = np.max(num_tokens)
# max_tokens = int(max_tokens)

# pad = "pre"

# x_train_pad = pad_sequences(
#     x_train_tokens, maxlen=max_tokens, padding=pad, truncating=pad
# )
# x_test_pad = pad_sequences(
#     x_test_tokens, maxlen=max_tokens, padding=pad, truncating=pad
# )


# idx = tokenizer.word_index
# inverse_map = dict(zip(idx.values(), idx.keys()))


# def tokens_to_string(tokens):
#     # Map from tokens back to words.
#     words = [inverse_map[token] for token in tokens if token != 0]

#     # Concatenate all words.
#     text = " ".join(words)

#     return text


# model = Sequential()


# embedding_size = 8

# model.add(
#     Embedding(
#         input_dim=num_words,
#         output_dim=embedding_size,
#         input_length=max_tokens,
#         name="layer_embedding",
#     )
# )

# model.add(GRU(units=16, return_sequences=True))
# model.add(GRU(units=8, return_sequences=True))
# model.add(GRU(units=4))

# # the first parameter corresponds to number of target
# model.add(
#     Dense(4, activation="softmax")
# )  # changed ativation to softmax and first parameter from one to 2 vhl
# optimizer = Adam(lr=1e-3)

# model.compile(
#     loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
# )  # changed loss to categorical_crossentropy

# # model.fit(x_train_pad, y_train, validation_split=0.05, epochs=3, batch_size=32)
# model.fit(
#     x_train_pad, one_hot_labels, validation_split=0.05, epochs=3, batch_size=32
# )  # replaced y_train with one_hot_labels
# result = model.evaluate(
#     x_test_pad, np.array(two_hot_labels)
# )  # replaced y_test with two_hot_abels

# print("Accuracy: {0:.2%}".format(result[1]))
# print("-----------------------------------------------------")

# y_pred = model.predict(x=x_test_pad[0:1000])
# y_pred = y_pred.T[0]

# cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
# cls_true = np.array(y_test[0:1000])

# incorrect = np.where(cls_pred != cls_true)
# idx = incorrect[0]
# print(idx)