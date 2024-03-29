import numpy as np
import tensorflow as tf
import talos
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.model_interface import ModelInterface
from datetime import datetime
from keras import backend as K


class LSTMQPredictor(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "LSTMQPredictor")

        self.input_shape = None
        self.train_model = None
        self.model = None
        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 6, 9, 12],
                               'first_conv_activation': ['relu'],
                               'second_lstm_dim': [16, 32, 64],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': ['relu'],
                               'batch_size': [256, 512, 1024],
                               'epochs': [4000],
                               'patience': [30],
                               'optimizer': ['adam'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }
        # self.q = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99])
        self.q = np.array([0.95])

    def training(self, X_train, y_train, X_test, y_test, p):
        training_start = datetime.now()
        history, self.model = self.talos_model(X_train, y_train, X_test, y_test, p)
        training_time = datetime.now() - training_start

        inference_start = datetime.now()
        prediction_quantiles = self.train_model.predict(X_test)
        inference_time = (datetime.now() - inference_start) / y_test.shape[0]

        return self.model, history, prediction_quantiles, training_time, inference_time

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

    def talos_model(self, X_train, y_train, x_val, y_val, p):
        tf.keras.backend.clear_session()
        input_shape = X_train.shape[1:]

        input_tensor = Input(shape=input_shape)

        x = Conv1D(filters=p['first_conv_dim'], kernel_size=p['first_conv_kernel'],
                   strides=1, padding="causal",
                   activation=p['first_conv_activation'],
                   input_shape=input_shape)(input_tensor)

        x = LSTM(p['second_lstm_dim'])(x)

        x = layers.Dense(p['first_dense_dim'], activation=p['first_dense_activation'])(x)

        outputs = layers.Dense(units=len(self.q))(x)

        self.train_model = Model(inputs=input_tensor, outputs=outputs)

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss=self.quantile_loss,
                                 optimizer=opt,
                                 metrics=["mse", "mae", self.quantile_loss])

        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        return history, self.model

    def quantile_loss(self, y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(self.q*e, (self.q-1)*e), axis=-1)

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return

        self.train_model.save_weights(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.h5',
                                      save_format="h5")
        self.count_save += 1
