import numpy as np
import tensorflow as tf
import talos
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.model_interface import ModelInterface
from datetime import datetime


class MCDLSTMPredictor(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "MCDLSTMPredictor")
        self.train_model = None
        self.input_shape = None
        self.model = None
        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 6, 9, 12],
                               'first_conv_activation': ['relu'],
                               'conv_dropout': [0.0, 0.05, 0.10],
                               'second_lstm_dim': [16, 32, 64],
                               'lstm_dropout': [0.0, 0.05, 0.10],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': ['relu'],
                               'dense_dropout': [0.0, 0.05, 0.10],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [30],
                               'optimizer': ['adam'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }

    def compute_predictions(self, X_test, iterations=30):
        prediction = []
        for i in range(iterations):
            prediction.append(self.model(X_test))
        print(len(prediction), prediction[0].shape)
        return np.mean(prediction), np.std(prediction)

    def training(self, X_train, y_train, X_test, y_test, p):
        training_start = datetime.now()
        history, self.model = self.talos_model(X_train, y_train, X_test, y_test, p)
        training_time = datetime.now() - training_start

        self.train_model.summary()
        print(history)

        inference_start = datetime.now()
        prediction_mean, prediction_std = self.compute_predictions(X_test)
        inference_time = (datetime.now() - inference_start) / y_test.shape[0]

        return self.model, history, prediction_mean, prediction_std, training_time, inference_time

    def tuning(self, X_tuning, y_tuning, p):
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        tuning_start = datetime.now()
        history = self.train_model.fit(X_tuning, y_tuning, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        tuning_time = datetime.now() - tuning_start
        self.model = save_check.dnn.model

        return self.model, history, tuning_time

    def training_talos(self, X_train, y_train, X_test, y_test, p):
        p = self.parameter_list
        tf.keras.backend.clear_session()
        self.input_shape = X_train.shape[1:]

        t = talos.Scan(x=X_train,
                       y=y_train,
                       model=self.talos_model,
                       experiment_name=self.name,
                       params=p,
                       clear_session=True,
                       print_params=True,
                       reduction_method='correlation',
                       reduction_metric='mse',
                       round_limit=250)

        return t, None, None

    def load_and_predict(self, X_train, y_train, X_test, y_test, p):
        self.train_model = self.load_model(X_train, p)
        self.model = self.train_model

        prediction_mean, prediction_std = self.compute_predictions(X_test)

        prediction_mean = np.concatenate(prediction_mean)
        prediction_std = np.concatenate(prediction_std)

        return self.model, prediction_mean, prediction_std

    def load_and_tune(self, X_train, y_train, X_test, y_test, p):
        global opt
        self.train_model = self.load_model(X_train, y_train, X_test, y_test, p)
        self.model = self.train_model

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        prediction_mean, prediction_std = self.compute_predictions(X_test)

        prediction_mean = np.concatenate(prediction_mean)
        prediction_std = np.concatenate(prediction_std)

        return self.model, prediction_mean, prediction_std

    def load_model(self, X_train, y_train, x_val, y_val, p):
        tf.keras.backend.clear_session()
        input_shape = X_train.shape[1:]

        self.train_model = Sequential([
            tf.keras.layers.Conv1D(filters=p['first_conv_dim'], kernel_size=p['first_conv_kernel'],
                                   strides=1, padding="causal",
                                   activation=p['first_conv_activation'],
                                   input_shape=input_shape),
            tf.keras.layers.Dropout(p['conv_dropout'], trainable=True),
            tf.keras.layers.LSTM(p['second_lstm_dim']),
            tf.keras.layers.Dropout(p['lstm_dropout'], trainable=True),
            tf.keras.layers.Dense(p['first_dense_dim'], activation=p['first_dense_activation']),
            tf.keras.layers.Dropout(p['dense_dropout'], trainable=True),
            tf.keras.layers.Dense(1),
        ])

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        self.train_model.load_weights(p['weight_file'])

        return self.train_model

    def talos_model(self, X_train, y_train, x_val, y_val, p):
        input_shape = X_train.shape[1:]

        self.train_model = Sequential([
            tf.keras.layers.Conv1D(filters=p['first_conv_dim'], kernel_size=p['first_conv_kernel'],
                                   strides=1, padding="causal",
                                   activation=p['first_conv_activation'],
                                   input_shape=input_shape),
            tf.keras.layers.Dropout(p['conv_dropout'], trainable=True),
            tf.keras.layers.LSTM(p['second_lstm_dim']),
            tf.keras.layers.Dropout(p['lstm_dropout'], trainable=True),
            tf.keras.layers.Dense(p['first_dense_dim'], activation=p['first_dense_activation']),
            tf.keras.layers.Dropout(p['dense_dropout'], trainable=True),
            tf.keras.layers.Dense(1),
        ])

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer=opt,
                                 metrics=["mse", "mae"])
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        return history, self.model

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.train_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf',
                              save_format="tf")
        self.count_save += 1
