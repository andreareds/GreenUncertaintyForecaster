# Performance and Energy Savings Trade-Off with Uncertainty-Aware Cloud Workload Forecasting

## Abstract

Cloud managers typically leverage future workload predictions to make informed decisions on resource allocation, where the ultimate goal of the allocation is to meet customers' demands while reducing the provisioning cost. Among several workload forecasting approaches proposed in the literature, uncertainty-aware time series analysis solutions are desirable in cloud scenarios because they can predict the distribution of future demand and provide bounds associated with a given service level set by the resource manager. The effectiveness of uncertainty-based workload predictions is normally assessed in terms of accuracy metrics (e.g.\ MAE) and service level (e.g.\ Success Rate), but the effect on the resource provisioning cost is under-investigated. 


We propose an evaluation framework to assess the impact of uncertainty-aware predictions on the performance vs cost trade-off, where we express the cost in terms of energy savings. We illustrate the framework's effectiveness by simulating two real-world cloud scenarios where an optimizer leverages workload predictions to allocate resources to satisfy a desired service level while minimizing energy waste. Offline experiments compare representative uncertainty-aware models and a new model (HBNN++) that we propose, which predict a cluster trace's GPU demand. We show that more effective uncertainty modelling can save energy without violating desired service level targets and that model performance varies depending on the specific details of the allocation scheme, server and GPU energy costs.

## Python Dependencies
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* statsmodels               0.12.2
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0
* tensorflow-probability    0.14.0

## Project Structure
* **hyperparams**: contains for each deep learning model the list of optimal hyperparameters found with Talos.
* **img**: contains output plot for predictions, models and loss function.
* **models**: contains the definition of statistical and deep learning models. One can train the model from scratch using the optimal parameters found with Talos, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **param**: contains for each statistical model the list of optimal parameters found.
* **res**: contains the results of the prediction
* **saved_data**: contains the preprocessed datasets.
* **saved_models**: contains the model saved during the training phase.
* **time**: contains measurements of the time for training, fine-tuning and inference phases.
* **util**: contains useful methods for initialising the datasets, plotting and saving the results.

## Statistical Methods

#### Train ARIMA

```bash
python arima_training.py
```

#### Train GARCH

```bash
python garch_training.py
```

## Deep Learning Methods

#### Train LSTM

```bash
python lstm_training.py
```

#### Train HBNN

```bash
python hbnn_training.py
```

#### Train LSTMD

```bash
python lstmd_training.py
```

