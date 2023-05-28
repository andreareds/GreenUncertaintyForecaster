import glob
import numpy as np
import os
import pandas as pd
import argparse
from models import FLBNN
from util import dataset, save_results


def main(args):
    wins = [eval(i) for i in args.windows.split("-")]
    hs = [eval(i) for i in args.horizons.split("-")]

    for win in wins:
        for res in args.resources.split("-"):
            for h in hs:
                for c in args.clusters.split("-"):
                    experiment_name = f"FLBNN-{c}-{res}-w{win}-h{h}"

                    # Data creation and load
                    ds = dataset.Dataset(args.datapath, meta=False, filename=f"ali20/{c}.csv", winSize=win, horizon=h,
                                         resource=res)
                    ds.dataset_creation()
                    ds.data_summary()
                    parameters = pd.read_csv(f"hyperparams/FLBNN-{c}-{res}-w{win}-h{h}.csv").iloc[0]

                    files = sorted(
                        glob.glob(f"saved_models/talos-FLBNN-{c}-{res}-w{win}-h{h}*_weights.tf.i*"))

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
                                train_model, _, prediction_mean, prediction_std, _, _ = model.training(ds.X_train,
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

                    train_distribution = train_model(ds.X_train)
                    train_mean = np.concatenate(train_distribution.mean().numpy(), axis=0)
                    train_std = np.concatenate(train_distribution.stddev().numpy(), axis=0)

                    save_results.save_uncertainty_csv(train_mean, train_std,
                                                      np.concatenate(ds.y_train, axis=0),
                                                      'avg' + res,
                                                      'train-' + model.name)

                    save_results.save_uncertainty_csv(np.concatenate(prediction_mean, axis=0),
                                                      np.concatenate(prediction_std, axis=0),
                                                      np.concatenate(ds.y_test[:len(prediction_mean)], axis=0),
                                                      'avg' + res,
                                                      model.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        default=False,
        type=str,
        required=True,
        help="The path to the data"
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main(parser.parse_args())
