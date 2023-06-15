import glob
import numpy as np
import os
import pandas as pd
import argparse
from models import FLBNN
from util import dataset, save_results
import tensorflow as tf

def main(args):
    #tf.config.experimental_run_functions_eagerly(True)
    wins = [eval(i) for i in args.windows.split("-")]
    hs = [eval(i) for i in args.horizons.split("-")]

    for win in wins:
        for res in args.resources.split("-"):
            for h in hs:
                for c in args.clusters.split("-"):
                    experiment_name = f"FLBNN-{c}-{res}-w{win}-h{h}"

                    # Data creation and load
                    ds = dataset.Dataset(f"{args.projectpath}/saved_data/", meta=False, filename=f"ali20/{c}.csv", winSize=win, horizon=h,
                                         resource=res)
                    ds.dataset_creation()
                    ds.data_summary()

                    # tf.compat.v1.disable_eager_execution()

                    files = []
                    p = None
                    if not args.tuning_hypers:
                        parameters = pd.read_csv(f"{args.projectpath}/hyperparams/FLBNN-{c}-{res}-w{win}-h{h}.csv").iloc[0]

                        files = sorted(
                            glob.glob(f"{args.projectpath}/saved_models/talos-FLBNN-{c}-{res}-w{win}-h{h}*_weights.tf.i*"))

                        dense_act = 'relu'
                        if 'relu' in parameters['first_dense_activation']:
                            dense_act = 'relu'
                        elif 'tanh' in parameters['first_dense_activation']:
                            dense_act = 'tanh'

                        p = {'first_conv_dim': parameters['first_conv_dim'],
                             'first_conv_kernel': (parameters['first_conv_kernel'],),
                             'first_conv_activation': parameters['first_conv_activation'],
                             'first_lstm_dim': parameters['first_lstm_dim'],
                             'first_dense_dim': parameters['first_dense_dim'],
                             'first_dense_activation': dense_act,
                             'batch_size': parameters['batch_size'],
                             'epochs': parameters['epochs'],
                             'patience': parameters['patience'],
                             'optimizer': parameters['optimizer'],
                             'batch_normalization': True,
                             'lr': parameters['lr'],
                             'decay': parameters['decay'],
                             }

                    print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                    model = FLBNN.FLBNNPredictor()
                    model.name = experiment_name
                    model.ds = ds  # TODO might want to pass a copy to be able to shuffle ds
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
                                train_model, _, prediction_mean, prediction_std, _, _ = model.training(ds.X_train,
                                                                                                       ds.y_train,
                                                                                                       ds.X_test,
                                                                                                       ds.y_test,
                                                                                                       p)
                    else:
                        if args.tuning_hypers:
                            train_model, _, _ = model.training_talos(ds.X_train,
                                                                     ds.y_train,
                                                                     ds.X_test,
                                                                     ds.y_test,
                                                                     p)
                        else:
                            train_model, _, prediction_mean, prediction_std, _, _ = model.training(ds.X_train,
                                                                                                   ds.y_train,
                                                                                                   ds.X_test,
                                                                                                   ds.y_test,
                                                                                                   p)

                    train_mean, train_std = model.compute_predictions(ds.X_train)

                    save_results.save_uncertainty_csv(args.projectpath,
                                                      train_mean, train_std,
                                                      np.concatenate(ds.y_train, axis=0),
                                                      'avg' + res,
                                                      'train-' + model.name)

                    save_results.save_uncertainty_csv(args.projectpath,
                                                      prediction_mean,
                                                      prediction_std,
                                                      np.concatenate(ds.y_test[:prediction_mean.shape[0]], axis=0),
                                                      'avg' + res,
                                                      model.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projectpath",
        default=False,
        type=str,
        required=True,
        help="The path to the project"
    )
    parser.add_argument(
        "--windows",
        default=False,
        type=str,
        required=True,
        help="The window sizes to use (provide integer numbers separated by dashes)."
    )
    parser.add_argument(
        "--horizons",
        default=False,
        type=str,
        required=True,
        help="The horizons to use (provide integer numbers separated by dashes)."
    )
    parser.add_argument(
        "--resources",
        default=False,
        type=str,
        required=True,
        help="The resources to use (provide names separated by dashes)."
    )
    parser.add_argument(
        "--clusters",
        default=None,
        type=str,
        required=True,
        help="The horizons to use (provide names separated by dashes)."
    )
    parser.add_argument(
        "--tuning_hypers",
        default=None,
        type=int,
        required=True,
        help="Whether to perform hyperparams tuning with talos (1 yes, 0 no)"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main(parser.parse_args())
    # imports are always needed

    tf.test.is_gpu_available() # True/False

    # Or only check for gpu's with cuda support
    tf.test.is_gpu_available(cuda_only=True)

