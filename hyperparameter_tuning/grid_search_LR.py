# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:22:36 2026

@author: USER
"""

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import random
import pickle
from utils import *

#%% Fix Seed

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)

SEED = 23 
set_seed(SEED)
#%% Load dataset / Preprocessing
dataset = pd.read_csv('./dataset/preprocessed_dataset_for_ML.csv')

#%% Dataset generation

X_train, X_test, y_train, y_test = categorization(dataset, SEED)
X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#%% Train final model

model = LinearRegression(n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_test_pred  = np.clip(model.predict(X_test_scaled),  0, 100)
y_train_pred = np.clip(model.predict(X_train_scaled), 0, 100)

train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test,  y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse  = root_mean_squared_error(y_test,  y_test_pred)

print(f"Train R2: {train_r2:.4f}  RMSE: {train_rmse:.4f}")
print(f"Test  R2: {test_r2:.4f}  RMSE: {test_rmse:.4f}")

#%% Parity Plot 
show_parity_plot(y_train, y_train_pred, y_test, y_test_pred)

#%% Save model / scaler

with open(f'./hyperparameter_tuning/output/lr_model_seed_{SEED}.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(f'./hyperparameter_tuning/output/lr_scaler_seed_{SEED}.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#%% Reproduce model

SEED = 23

with open(f'./hyperparameter_tuning/output/lr_model_seed_{SEED}.pkl',  'rb') as f:
    model = pickle.load(f)

with open(f'./hyperparameter_tuning/output/lr_scaler_seed_{SEED}.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

y_test_pred  = np.clip(model.predict(X_test_scaled),  0, 100)
y_train_pred = np.clip(model.predict(X_train_scaled), 0, 100)

train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test,  y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse  = root_mean_squared_error(y_test,  y_test_pred)

print(f"Train R2: {train_r2:.4f}  RMSE: {train_rmse:.4f}")
print(f"Test  R2: {test_r2:.4f}  RMSE: {test_rmse:.4f}")
