import glob
import numpy as np
import os
import pandas as pd
from keras.utils.vis_utils import plot_model
from models import FLBNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from util import dataset, plot_training, save_results


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    wins = [144]  # TODO what is wins?
    hs = [2]
    # resources = ['cpu', 'mem']
    # clusters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    resources = ['cpu']
    clusters = ['a']

    for win in wins:
        for res in resources:
            for h in hs:
                for c in clusters:
                    mses, maes = [], []  # TODO to remove
                    experiment_name = 'HBNN-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                    # Data creation and load
                    ds = dataset.Dataset(meta=False, filename='res_task_' + c + '.csv', winSize=win, horizon=h,
                                         resource=res)
                    ds.dataset_creation()
                    ds.data_summary()
                    parameters = pd.read_csv("hyperparams/p_flbnn-" + c + ".csv").iloc[0]

                    # TODO why this is always cpu??
                    files = sorted(
                        glob.glob("saved_models/talos-FLBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                    dense_act = 'relu'
                    if 'relu' in parameters['first_dense_activation']:
                        dense_act = 'relu'
                    elif 'tanh' in parameters['first_dense_activation']:
                        dense_act = 'tanh'

                    p = {'first_conv_dim': parameters['first_conv_dim'],
                         'first_conv_kernel': (parameters['first_conv_kernel'],),
                         'first_conv_activation': parameters['first_conv_activation'],
                         'first_lstm_dim': parameters['second_lstm_dim'],
                         'first_dense_dim': parameters['first_dense_dim'],
                         'first_dense_activation': dense_act,
                         'batch_size': parameters['batch_size'],
                         'epochs': parameters['epochs'],
                         'patience': parameters['patience'],
                         'optimizer': parameters['optimizer'],
                         'batch_normalization': True,
                         'lr': parameters['lr'],
                         'momentum': parameters['momentum'],
                         'decay': parameters['decay'],
                         'pred_steps': 0,
                         }

                    print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                    model = FLBNN.FLBNNPredictor()
                    model.name = experiment_name
                    train_model = None
                    prediction_mean = None
                    prediction_std = None

                    if len(files):
                        for i in range(len(files)):
                            path_weight = files[-(i + 1)][:-6]
                            p['weight_file'] = path_weight
                            # TODO dont like this try except
                            try:
                                train_model, prediction_mean, prediction_std = model.load_and_predict(ds.X_train,
                                                                                                      ds.y_train,
                                                                                                      ds.X_test,
                                                                                                      ds.y_test,
                                                                                                      p)
                            except:
                                train_model, prediction_mean, prediction_std = model.training(ds.X_train,
                                                                                              ds.y_train,
                                                                                              ds.X_test,
                                                                                              ds.y_test, p)
                    else:
                        train_model, prediction_mean, prediction_std = model.training(ds.X_train,
                                                                                      ds.y_train,
                                                                                      ds.X_test,
                                                                                      ds.y_test,
                                                                                      p)

                    train_distribution = train_model(ds.X_train)
                    train_mean = np.concatenate(train_distribution.mean().numpy(), axis=0)
                    train_std = np.concatenate(train_distribution.stddev().numpy(), axis=0)

                    save_results.save_uncertainty_csv(train_mean, train_std,
                                                      np.concatenate(ds.y_train, axis=0),
                                                      'avg' + res,
                                                      'train-' + model.name)

                    # TODO why you don't use these values?
                    mse = mean_squared_error(ds.y_test, prediction_mean)
                    mae = mean_absolute_error(ds.y_test, prediction_mean)

                    save_results.save_uncertainty_csv(prediction_mean, prediction_std,
                                                      np.concatenate(ds.y_test[:len(prediction_mean)], axis=0),
                                                      'avg' + res,
                                                      model.name)

                    plot_training.plot_series_interval(np.arange(0, len(ds.y_test) - 1), ds.y_test, prediction_mean,
                                                       prediction_std,
                                                       label1="ground truth",
                                                       label2="prediction", title=model.name)

                    plot_model(train_model, to_file='img/models/model_plot_' + model.name + '.png', show_shapes=True,
                               show_layer_names=True)


if __name__ == "__main__":
    main()
