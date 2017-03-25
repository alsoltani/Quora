import numpy as np
import pandas as pd
import h5py
import cPickle as pickle

from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K


def create_model():
    word_index = pickle.load(open("word_index.p", "rb"))

    max_nb_words = 200000  # Maximum size of the vocabulary
    max_seq_length = 40  # Number of words from the question to be used
    embedding_dim = 300  # Dimension of the word embeddings
    nb_words = min(max_nb_words, len(word_index))

    embedding_matrix = np.load("embedding_matrix.npy")
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                name="embedding",
                                input_length=max_seq_length)
    inputs = {}
    pre_merge = {}

    for i in xrange(6):
        inputs[i] = Input(shape=(max_seq_length,), name="input_{}".format(i + 1), dtype='int32')
        pre_merge[i] = embedding_layer(inputs[i])

    for i in xrange(2):
        pre_merge[i] = TimeDistributed(Dense(embedding_dim, activation='relu'))(pre_merge[i])
        pre_merge[i] = Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,))(pre_merge[i])

    for i in xrange(2, 4):
        filter_length = 5
        nb_filter = 64
        pool_length = 4

        pre_merge[i] = Convolution1D(nb_filter=nb_filter,
                                     filter_length=filter_length,
                                     border_mode='valid',
                                     activation='relu',
                                     subsample_length=1)(pre_merge[i])

        pre_merge[i] = Dropout(0.2)(pre_merge[i])
        pre_merge[i] = Convolution1D(nb_filter=nb_filter,
                                     filter_length=filter_length,
                                     border_mode='valid',
                                     activation='relu',
                                     subsample_length=1)(pre_merge[i])

        pre_merge[i] = GlobalMaxPooling1D()(pre_merge[i])

        pre_merge[i] = Dropout(0.2)(pre_merge[i])
        pre_merge[i] = Dense(300)(pre_merge[i])
        pre_merge[i] = Dropout(0.2)(pre_merge[i])

        pre_merge[i] = BatchNormalization()(pre_merge[i])

    for i in xrange(4, 6):
        pre_merge[i] = LSTM(300, dropout_W=0.2, dropout_U=0.2)(pre_merge[i])

    merged = merge(list(pre_merge.values()), mode='concat')
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    output = Dense(1, activation='sigmoid', name="output")(merged)

    model = Model(input=list(inputs.values()), output=output)
    model.get_layer("embedding").trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])  # 'precision', 'recall', 'fbeta_score'

    return model


def train(model, h_file, callbacks):
    model.fit({"input_1": h_file["train/data_1"], "input_2": h_file["train/data_2"],
               "input_3": h_file["train/data_1"], "input_4": h_file["train/data_2"],
               "input_5": h_file["train/data_1"], "input_6": h_file["train/data_2"]
               },
              {"output": h_file["train/labels"]},

              class_weight="auto",
              callbacks=callbacks,
              validation_split=0.1, nb_epoch=25, batch_size=512, shuffle=True)


def predict(model, h_file, weights=None, save_to=None):
    if weights is not None:
        model.load_weights(weights)

    predictions = \
        model.predict({"input_1": h_file["test/data_1"], "input_2": h_file["test/data_2"],
                       "input_3": h_file["test/data_1"], "input_4": h_file["test/data_2"],
                       "input_5": h_file["test/data_1"], "input_6": h_file["test/data_2"]
                       },
                      batch_size=512)

    if save_to is not None:
        submission = pd.DataFrame({"test_id": h_file['test/id'], "is_duplicate": predictions.ravel()})
        submission.to_csv(save_to, index=False)


if __name__ == "__main__":
    m = create_model()
    h = h5py.File("data.h5", "r")
    c = [ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)]

    train(m, h, callbacks=c)
    h.close()
