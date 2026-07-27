#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import random
from base_model.config import base_params
from utils import *

#%% Fix Seed

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
SEED = 23 
set_seed(SEED)

#%% Clean name

def clean_names(df):
    new_cols = []
    for col in df.columns:

        name = col
        name = name.replace('(', '').replace(')', '').replace('/', '').replace('[', '').replace(']', '').replace('-', '_').replace(',', '_')
        name = name.replace(' ', '_')
        new_cols.append(name)
    df.columns = new_cols
    return df

#%% Remove JSON invalid characters
X_train, X_test, y_train, y_test = categorization(dataset, SEED)
X_train = clean_names(X_train)
X_test  = clean_names(X_test)

#%% K-fold validation
set_seed(SEED)
params = base_params['LightGBM']
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = LGBMRegressor(**params)
    
    model.fit(X_tr, y_tr, 
              eval_set=[(X_val, y_val)],
              eval_metric='rmse')

    y_val_pred = model.predict(X_val)   
    y_val_pred = np.clip(y_val_pred, 0, 100)
    
    r2 = r2_score(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)
    
    best_iter = model.best_iteration_
    best_iterations.append(best_iter) 

    print(f"Fold: {fold + 1} - R2 score: {r2:.4f} - RMSE: {rmse:.4f}")
    print(f"Best Iteration: {best_iter}")

#%% Re-train model

avg_iteration = int(np.mean(best_iterations))

model = LGBMRegressor(random_state=SEED,  n_jobs=-1,  n_estimators=avg_iteration,  verbosity=-1)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)   
y_train_pred = model.predict(X_train)

y_test_pred = np.clip(y_test_pred, 0, 100)
y_train_pred = np.clip(y_train_pred, 0, 100)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print(f"Train R2 score: {train_r2:.4f} - RMSE: {train_rmse:.4f}")
print(f"Test R2 score:  {test_r2:.4f} - RMSE: {test_rmse:.4f}")

#%% Parity Plot 
show_parity_plot(y_train, y_train_pred, y_test, y_test_pred)

#%% Save model
model.booster_.save_model(f"./base_model/output/lightGBM_model_seed{SEED}.txt")

#%% Load model
loaded_model = lgb.Booster(model_file=f"./base_model/output/lightGBM_model_seed{SEED}.txt")

y_test_pred = model.predict(X_test)   
y_train_pred = model.predict(X_train)

y_test_pred = np.clip(y_test_pred, 0, 100)
y_train_pred = np.clip(y_train_pred, 0, 100)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
print(f"Train R2 score: {train_r2:.4f} - RMSE: {train_rmse:.4f}")
print(f"Test R2 score:  {test_r2:.4f} - RMSE: {test_rmse:.4f}")