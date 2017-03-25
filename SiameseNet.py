from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import h5py
import cPickle as pickle

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Embedding, merge, Bidirectional
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2


def l2_norm(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def l1_norm(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=1, keepdims=True)


def norm_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def load_embeddings():
    word_index = pickle.load(open("word_index.p", "rb"))
    max_nb_words = 200000  # Maximum size of the vocabulary
    max_sequence_length = 40  # Number of words from the question to be used

    number_words = min(max_nb_words, len(word_index))
    embedding_matrix = np.load("embedding_matrix.npy")

    return embedding_matrix, number_words, max_sequence_length


def create_base_network(input_dim, embedding_matrix, max_sequence_length):
    """
    Base network to be shared (eq. to feature extraction).
    """

    seq = Sequential()
    seq.add(Embedding(input_dim,
                      embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      name="embedding",
                      input_length=max_sequence_length,
                      embeddings_regularizer=l2(0.01),
                      trainable=False))
    seq.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True)))
    seq.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2)))
    return seq


def create_upper_network(max_sequence_length):
    input_a = Input(shape=(max_sequence_length,), name="input_1", dtype='int32')
    input_b = Input(shape=(max_sequence_length,), name="input_2", dtype='int32')

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    l1 = Lambda(l1_norm, output_shape=norm_output_shape)([processed_a, processed_b])
    l2 = Lambda(l2_norm, output_shape=norm_output_shape)([processed_a, processed_b])

    angle = merge([processed_a, processed_b], mode="mul")
    combined = merge([l1, l2, angle], mode="concat")

    fc = Dense(300, init="glorot_normal")(combined)
    fc = BatchNormalization()(fc)
    fc = PReLU()(fc)
    fc = Dropout(0.2)(fc)

    fc = Dense(300, init="glorot_normal")(fc)
    fc = BatchNormalization()(fc)
    fc = PReLU()(fc)
    fc = Dropout(0.2)(fc)

    output = Dense(1, activation='sigmoid', name="output")(fc)

    model = Model([input_a, input_b], output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    return model


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy with a fixed threshold on distances.
    """
    return labels[predictions.ravel() < 0.5].mean()


if __name__ == "__main__":
    embed_matrix, nb_words, max_seq_length = load_embeddings()

    # network definition
    base_network = create_base_network(nb_words, embed_matrix, max_seq_length)
    m = create_upper_network(max_seq_length)

    # train
    h_file = h5py.File("data.h5", "r")
    callbacks = [ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)]

    m.fit({"input_1": h_file["train/data_1"],
           "input_2": h_file["train/data_2"]
           },
          {"output": h_file["train/labels"]},

          class_weight="auto",
          callbacks=callbacks,
          validation_split=0.1, nb_epoch=200, batch_size=512, shuffle=True)
