import h5py
import sys
import tensorflow as tf
import numpy as np
import argparse
import glob
import csv

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, Callback, ModelCheckpoint
from sklearn.metrics import f1_score


class ComputeTestF1(Callback):
    """Custom callback to calculate F1 score"""
    def on_epoch_end(self, epochs, logs):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_target = self.validation_data[1]
        logs['val_f1'] = f1_score(val_target, val_predict)


def get_args():
    parser = argparse.ArgumentParser(description='Run probing experiment for c2f-coref with BERT embeddings')
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--ablate_boundary', action='store_true')
    parser.add_argument('--ablate_attention', action='store_true')
    parser.add_argument('--ablate_span_width', action='store_true')
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    log_name = args.exp_name + '.tsv'
    filenames = glob.glob(args.train_data + "/*.h5")
    train_data = []
    test_data_flag = True if args.test_data is not None else False

    for fn in filenames:
        with h5py.File(fn, 'r') as f:
            train_data.append(f.get('span_representations').value)
    train_data = np.concatenate(train_data, axis=0)
    x_train = train_data[:, :-2]
    y_train = train_data[:, -1].astype(int)

    # test random label
    # y_train = np.random.randint(2, size=y_train.shape)

    if (args.random):
        reps_pool = np.concatenate((x_train[:, :3*args.embed_dim + 20], x_train[:, 3*args.embed_dim + 20:]), axis=0)
        sampled_reps_pool = reps_pool[np.random.choice(reps_pool.shape[0], size = y_train.shape[0], replace=True), :]

        train_parent_span, train_child_span = x_train[:, :3*args.embed_dim + 20], x_train[:, 3*args.embed_dim + 20:]
        rand_indices = np.random.randint(2, size=y_train.shape)
        for i in range(train_parent_span.shape[0]):
            if rand_indices[i] == 0:
                train_parent_span[i, :] = sampled_reps_pool[i, :]
            else:
                train_child_span[i, :] = sampled_reps_pool[i, :]
        x_train = np.concatenate((train_parent_span, train_child_span), axis=1)

    with h5py.File(args.val_data, 'r') as f:
        val_data = f.get('span_representations').value
        x_val = val_data[:, :-2]
        y_val = val_data[:, -1].astype(int)

    if test_data_flag:
        with h5py.File(args.test_data, 'r') as f:
            test_data = f.get('span_representations').value
            x_test = test_data[:, :-2]
            y_test = test_data[:, -1].astype(int)

    if (args.ablate_boundary):
        train_parent_span, train_child_span = x_train[:, :3*args.embed_dim + 20], x_train[:, 3*args.embed_dim + 20:]
        x_train_parent, x_train_child = train_parent_span[:, 2*args.embed_dim:], train_child_span[:, 2*args.embed_dim:]
        x_train = np.concatenate((x_train_parent, x_train_child), axis=1)

        val_parent_span, val_child_span = x_val[:, :3*args.embed_dim + 20], x_val[:, 3*args.embed_dim + 20:]
        x_val_parent, x_val_child = val_parent_span[:, 2*args.embed_dim:], val_child_span[:, 2*args.embed_dim:]
        x_val = np.concatenate((x_val_parent, x_val_child), axis=1)
        if test_data_flag:
            test_parent_span, test_child_span = x_test[:, :3*args.embed_dim + 20], x_test[:, 3*args.embed_dim + 20:]
            x_test_parent, x_test_child = test_parent_span[:, 2*args.embed_dim:], test_child_span[:, 2*args.embed_dim:]
            x_test = np.concatenate((x_test_parent, x_test_child), axis=1)
        print("Ablate boundary representations")
        print(x_train.shape)
    elif (args.ablate_attention):
        train_parent_span, train_child_span = x_train[:, :3*args.embed_dim + 20], x_train[:, 3*args.embed_dim + 20:]
        x_train_parent = np.delete(train_parent_span, np.s_[2*args.embed_dim:-20], axis=1)
        x_train_child = np.delete(train_child_span, np.s_[2*args.embed_dim:-20], axis=1)
        x_train = np.concatenate((x_train_parent, x_train_child), axis=1)

        val_parent_span, val_child_span = x_val[:, :3*args.embed_dim + 20], x_val[:, 3*args.embed_dim + 20:]
        x_val_parent = np.delete(val_parent_span, np.s_[2*args.embed_dim:-20], axis=1)
        x_val_child = np.delete(val_child_span, np.s_[2*args.embed_dim:-20], axis=1)
        x_val = np.concatenate((x_val_parent, x_val_child), axis=1)
        if test_data_flag:
            test_parent_span, test_child_span = x_test[:, :3*args.embed_dim + 20], x_test[:, 3*args.embed_dim + 20:]
            x_test_parent = np.delete(test_parent_span, np.s_[2*args.embed_dim:-20], axis=1)
            x_test_child = np.delete(test_child_span, np.s_[2*args.embed_dim:-20], axis=1)
            x_test = np.concatenate((x_test_parent, x_test_child), axis=1)
        print("Ablate attentional heads")
        print(x_train.shape)
    elif (args.ablate_span_width):
        train_parent_span, train_child_span = x_train[:, :3*args.embed_dim + 20], x_train[:, 3*args.embed_dim + 20:]
        x_train_parent, x_train_child = train_parent_span[:, :-20], train_child_span[:, :-20]
        x_train = np.concatenate((x_train_parent, x_train_child), axis=1)

        val_parent_span, val_child_span = x_val[:, :3*args.embed_dim + 20], x_val[:, 3*args.embed_dim + 20:]
        x_val_parent, x_val_child = val_parent_span[:, :-20], val_child_span[:, :-20]
        x_val = np.concatenate((x_val_parent, x_val_child), axis=1)
        if test_data_flag:
            test_parent_span, test_child_span = x_test[:, :3*args.embed_dim + 20], x_test[:, 3*args.embed_dim + 20:]
            x_test_parent, x_test_child = test_parent_span[:, :-20], test_child_span[:, :-20]
            x_test = np.concatenate((x_test_parent, x_test_child), axis=1)
        print("Ablate span width embeddings")
        print(x_train.shape)
    else:
        print("Using all features")

    # Probing model implementation using keras, following hyperparameters described in Liu's paper, can finetune later.
    model = Sequential()
    model.add(Dense(units=1024, activation='relu', input_dim=x_train.shape[1], use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')) # still need to check whether 1024 is correct, little details about this.
    model.add(Dense(units=1, activation='sigmoid'))
    opt = optimizers.Adam(lr=0.001)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', restore_best_weights=True)
    checkpoint_save = ModelCheckpoint(args.exp_name + '.h5', save_best_only=True, monitor='val_loss', mode='min')
    callbacks = [early_stop, checkpoint_save, ComputeTestF1(), CSVLogger(log_name, separator='\t')]

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val), callbacks=callbacks)

    with open(log_name, 'a') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        val_loss_and_metrics = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
        val_predict = (np.asarray(model.predict(x_val))).round()
        val_target = y_val
        best_val_f1 = f1_score(val_target, val_predict)
        tsv_writer.writerow(['best_val_acc', val_loss_and_metrics[1]])
        tsv_writer.writerow(['best_val_f1', best_val_f1])

        if test_data_flag:
            test_loss_and_metrics = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
            test_predict = (np.asarray(model.predict(x_test))).round()
            test_target = y_test
            best_test_f1 = f1_score(test_target, test_predict)
            tsv_writer.writerow(['best_test_acc', test_loss_and_metrics[1]])
            tsv_writer.writerow(['best_test_f1', best_test_f1])
