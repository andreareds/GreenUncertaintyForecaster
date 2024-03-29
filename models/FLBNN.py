import numpy as np
import tensorflow as tf
from tensorflow import keras
import talos
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.model_interface import ModelInterface
import tensorflow_probability as tfp
from datetime import datetime
import keras.backend as K


@tf.keras.utils.register_keras_serializable()
class VarLayer(tfp.layers.DenseVariational):
    def __init__(self, name, units, make_prior_fn, make_posterior_fn, kl_weight, activation, **kwargs):
        super().__init__(units=units, make_prior_fn=make_prior_fn, make_posterior_fn=make_posterior_fn, name=name,
                         kl_weight=kl_weight, activation=activation, **kwargs)

    def get_config(self):
        config = super(VarLayer, self).get_config()
        config.update({
            'name': self.name,
            'units': self.units,
            'activation': self.activation})
        return config

    def call(self, inputs):
        return super(VarLayer, self).call(inputs)


@tf.keras.utils.register_keras_serializable()
class VarCNNLayer(tfp.layers.Convolution1DReparameterization):
    def __init__(self, name, filters, kernel_size, strides, padding, activation, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, name=name,
                         padding=padding, activation=activation, kernel_divergence_fn=None, **kwargs)

    def get_config(self):
        config = super(VarCNNLayer, self).get_config()
        config.update({
            'name': self.name,
            'filters': self.filters,
            'activation': self.activation})
        return config

    def call(self, inputs):
        return super(VarCNNLayer, self).call(inputs)


class FLBNNPredictor(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "FLBNNPredictor")
        self.input_shape = None
        self.train_model = None
        self.model = None
        self.ds = None

        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 6, 9, 12],
                               'first_conv_activation': ['relu'],
                               'first_lstm_dim': [16, 32, 64],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': [keras.activations.relu],
                               'batch_size': [256, 512, 1024],
                               'epochs': [3000],
                               'patience': [30],
                               'optimizer': ['adam'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }

    def compute_predictions(self, X_test, iterations=30):
        predictions = [np.concatenate(self.model(X_test).numpy(), axis=0) for i in range(iterations)]
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def training(self, X_train, y_train, X_test, y_test, p):
        # self.build_batches(p['batch_size'])
        training_start = datetime.now()
        history, self.model = self.talos_model(X_train, y_train, X_test, y_test, p)
        training_time = datetime.now() - training_start

        inference_start = datetime.now()
        prediction_mean, prediction_std = self.compute_predictions(X_test)
        inference_time = (datetime.now() - inference_start) / y_test.shape[0]

        return self.model, history, prediction_mean, prediction_std, training_time, inference_time

    def load_and_predict(self, X_train, y_train, X_test, y_test, p):
        self.train_model = self.load_model(X_train, p)
        self.model = self.train_model

        prediction_mean, prediction_std = self.compute_predictions(X_test)

        prediction_mean = np.concatenate(prediction_mean)
        prediction_std = np.concatenate(prediction_std)

        return self.model, prediction_mean, prediction_std

    def load_and_tune(self, X_train, y_train, X_test, y_test, p):
        self.train_model = self.load_model(X_train, p)
        self.model = self.train_model
        opt = None

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss=self.negative_loglikelihood,
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=self.ds.X_train.shape[0],
                             validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        prediction_mean, prediction_std = self.compute_predictions(X_test)

        prediction_mean = np.concatenate(prediction_mean)
        prediction_std = np.concatenate(prediction_std)

        return self.model, prediction_mean, prediction_std

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
        tf.keras.backend.clear_session()
        self.input_shape = X_train.shape[1:]

        t = talos.Scan(x=X_train,
                       y=y_train,
                       model=self.talos_model,
                       experiment_name=self.name,
                       params=self.parameter_list,
                       clear_session=True,
                       print_params=True,
                       reduction_method='correlation',
                       reduction_metric="mae",
                       round_limit=250)

        return t, None, None

    def load_model(self, X_train, p):

        tf.keras.backend.clear_session()
        input_shape = X_train.shape[1:]

        input_tensor = Input(shape=input_shape)

        # Bayesian 1DCNN
        x = tfp.layers.Convolution1DReparameterization(
            filters=p['first_conv_dim'],
            kernel_size=p['first_conv_kernel'],
            strides=1,
            padding="valid",
            activation=p['first_conv_activation'],
        )(input_tensor)

        # LSTM
        x = LSTM(p['first_lstm_dim'])(x)

        # # Dense (optional)
        # x = layers.Dense(units=p['first_dense_dim'])(x)

        # Bayesian
        x = tfp.layers.DenseVariational(name='var',
                                        units=p['first_dense_dim'],
                                        make_prior_fn=self.prior,
                                        make_posterior_fn=self.posterior,
                                        kl_weight=1 / X_train.shape[0],
                                        activation=p['first_dense_activation'],
                                        )(x)

        outputs = layers.Dense(units=1)(x)

        opt = None
        self.train_model = Model(inputs=input_tensor, outputs=outputs)

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss=self.negative_loglikelihood,
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        self.train_model.load_weights(p['weight_file'])

        return self.train_model

    def talos_model(self, X_train, y_train, x_val, y_val, p):
        tf.keras.backend.clear_session()
        input_shape = X_train.shape[1:]

        input_tensor = Input(shape=input_shape)

        # Bayesian 1DCNN
        # x = tfp.layers.Convolution1DReparameterization(
        #     filters=p['first_conv_dim'],
        #     kernel_size=p['first_conv_kernel'],
        #     strides=1,
        #     padding="valid",
        #     activation=p['first_conv_activation'],
        # )(input_tensor)

        # Bayesian 1DCNN
        x = VarCNNLayer('varcnn',
                        p['first_conv_dim'],
                        p['first_conv_kernel'],
                        1,
                        "valid",
                        p['first_conv_activation'],
                        )(input_tensor)

        # LSTM
        x = LSTM(p['first_lstm_dim'])(x)

        # # Dense (optional)
        # x = layers.Dense(units=p['first_dense_dim'])(x)

        # Bayesian
        x = VarLayer('var', p['first_dense_dim'],
                     self.prior,
                     self.posterior,
                     1 / X_train.shape[0],
                     p['first_dense_activation'])(x)

        outputs = layers.Dense(units=1)(x)
        # outputs = tfp.layers.DenseReparameterization(1)(x)
        opt = None
        self.train_model = Model(inputs=input_tensor, outputs=outputs)

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])

        # self.train_model.compile(loss=self.negative_loglikelihood,
        #                          optimizer=opt,
        #                          metrics=["mse", "mae"])

        self.train_model.compile(loss='mean_absolute_error',
                                 optimizer=opt,
                                 #run_eagerly=True,
                                 metrics=["mse", "mae"])

        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        # history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=self.ds.X_train.shape[0],
        #                                validation_split=0.2, verbose=2, callbacks=[es, save_check])

        # self.train_model.build(input_shape)

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        return history, self.model

    def negative_loglikelihood(self, y_true, y_pred):
        predictions = [self.train_model(self.ds.X_train) for i in range(10)]

        y_pred_mean, y_pred_log_sigma = tf.math.reduce_mean(predictions, axis=0), \
                                        tf.math.reduce_std(predictions, axis=0)

        # Calculate the negative log likelihood
        loss = 0.5 * K.log(2 * np.pi) + y_pred_log_sigma + 0.5 * \
               K.square((self.ds.y_train - y_pred_mean) / K.exp(y_pred_log_sigma))
        return K.mean(loss)

    @staticmethod
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    @staticmethod
    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n)),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    def build_batches(self, batch_size):
        for i in range(0, int(len(self.ds.X_train) / batch_size), batch_size):
            if i + batch_size > len(self.ds.X_train):
                j = len(self.ds.X_train)
            else:
                j = i + batch_size
            self.X_batches[i] = self.ds.X_train[i:j]
            self.y_batches[i] = self.ds.y_train[i:j]

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return

        self.train_model.save_weights(self.model_path + self.name + str(self.count_save).zfill(4) + '_weights.tf',
                                      save_format="tf")

        self.count_save += 1
