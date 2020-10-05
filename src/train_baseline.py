import h5py
import sys
import tensorflow as tf
import numpy as np
import argparse
import glob
import keras
import csv

from keras.models import Model
from keras.layers import Dense, Conv1D, Input
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, Callback, ModelCheckpoint
from sklearn.metrics import f1_score
from keras_self_attention import SeqWeightedAttention

MAX_SPAN_WIDTH = 30


class ComputeTestF1(Callback):
    """Custom callback to calculate F1 score"""
    def on_epoch_end(self, epochs, logs):
        val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        val_target = self.validation_data[2]
        logs['val_f1'] = f1_score(val_target, val_predict)


def get_args():
    parser = argparse.ArgumentParser(description='Run baseline probing experiment for c2f-coref with BERT embeddings')
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--cnn_context', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    log_name = args.exp_name + '.tsv'
    embed_dim = proj_dim = args.embed_dim
    filenames = glob.glob(args.train_data + "/*.h5")
    train_data = []
    test_data_flag = True if args.test_data is not None else False
    
    for fn in filenames:
        with h5py.File(fn, 'r') as f:
            train_data.append(f.get('span_representations').value)
    train_data = np.concatenate(train_data, axis=0)
    x_train = train_data[:, :-2]
    y_train = train_data[:, -1].astype(int)
    train_parent_emb = x_train[:, :MAX_SPAN_WIDTH*embed_dim].reshape(x_train.shape[0], MAX_SPAN_WIDTH, embed_dim)
    train_child_emb = x_train[:, MAX_SPAN_WIDTH*embed_dim:].reshape(x_train.shape[0], MAX_SPAN_WIDTH, embed_dim)

    with h5py.File(args.val_data, 'r') as f:
        val_data = f.get('span_representations').value
        x_val = val_data[:, :-2]
        y_val = val_data[:, -1].astype(int)
        val_parent_emb = x_val[:, :MAX_SPAN_WIDTH*embed_dim].reshape(x_val.shape[0], MAX_SPAN_WIDTH, embed_dim)
        val_child_emb = x_val[:, MAX_SPAN_WIDTH*embed_dim:].reshape(x_val.shape[0], MAX_SPAN_WIDTH, embed_dim)

    if test_data_flag:
        with h5py.File(args.test_data, 'r') as f:
            test_data = f.get('span_representations').value
            x_test = test_data[:, :-2]
            y_test = test_data[:, -1].astype(int)
            test_parent_emb = x_test[:, :MAX_SPAN_WIDTH*embed_dim].reshape(x_test.shape[0], MAX_SPAN_WIDTH, embed_dim)
            test_child_emb = x_test[:, MAX_SPAN_WIDTH*embed_dim:].reshape(x_test.shape[0], MAX_SPAN_WIDTH, embed_dim)

    # Input is of [batch_size, max_span_width, embed_dim]
    # train_baseline_model(x_train, y_train, x_val, y_val, x_test, y_test, kernel_size=3)

    k = 1 + 2*args.cnn_context
    parent_span = Input(shape=(MAX_SPAN_WIDTH, embed_dim))
    child_span = Input(shape=(MAX_SPAN_WIDTH, embed_dim))

    # Shared CNN for parent and child span representations as a projection of local context
    cnn_projection = Conv1D(proj_dim, kernel_size=k, strides=1, padding='same', kernel_initializer='he_normal',input_shape=(MAX_SPAN_WIDTH, embed_dim))
    encoded_parent_span = cnn_projection(parent_span)  # [batch_size, max_span_width, proj_dim]
    encoded_child_span = cnn_projection(child_span)  # [batch_size, max_span_width, proj_dim]

    # Shared Attention for parent and child mention span representations
    attention = SeqWeightedAttention()
    headwords_1 = attention(encoded_parent_span)  # [batch_size, embed_dim]
    headwords_2 = attention(encoded_child_span)  # [batch_size, embed_dim]

    headwords_vector = keras.layers.concatenate([headwords_1, headwords_2], axis=1)
    hidden_layer = Dense(units=1024, activation='relu', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(headwords_vector)
    predictions = Dense(units=1, activation='sigmoid')(hidden_layer)

    opt = optimizers.Adam(lr=0.001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', restore_best_weights=True)
    checkpoint_save = ModelCheckpoint(args.exp_name + '.h5', save_best_only=True, monitor='val_loss', mode='min')
    callbacks = [early_stop, checkpoint_save, ComputeTestF1(), CSVLogger(log_name, separator='\t')]

    model = Model(inputs=[parent_span, child_span], output=predictions)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit([train_parent_emb, train_child_emb], y_train, epochs=50, batch_size=512, validation_data=([val_parent_emb, val_child_emb], y_val), callbacks=callbacks)

    with open(log_name, 'a') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        val_loss_and_metrics = model.evaluate([val_parent_emb, val_child_emb], y_val, batch_size=x_val.shape[0])
        val_predict = (np.asarray(model.predict([val_parent_emb, val_child_emb]))).round()
        val_target = y_val
        best_val_f1 = f1_score(val_target, val_predict)
        tsv_writer.writerow(['best_val_acc', val_loss_and_metrics[1]])
        tsv_writer.writerow(['best_val_f1', best_val_f1])

        if test_data_flag:
            test_loss_and_metrics = model.evaluate([test_parent_emb, test_child_emb], y_test, batch_size=x_test.shape[0])
            test_predict = (np.asarray(model.predict([test_parent_emb, test_child_emb]))).round()
            test_target = y_test
            best_test_f1 = f1_score(test_target, test_predict)
            tsv_writer.writerow(['best_test_acc', test_loss_and_metrics[1]])
            tsv_writer.writerow(['best_test_f1', best_test_f1])
