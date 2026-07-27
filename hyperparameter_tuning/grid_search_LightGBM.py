# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:16:36 2026

@author: USER
"""

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import random
import pickle
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

#%% Load dataset / Preprocessing
dataset = pd.read_csv('./dataset/preprocessed_dataset_for_ML.csv')

#%% Remove JSON invalid characters
X_train, X_test, y_train, y_test = categorization(dataset, SEED)
X_train = clean_names(X_train)
X_test  = clean_names(X_test)

#%% Perform grid search
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "n_estimators": [1000, 2000, 3000, 5000]
}

results = []

# ===============================
# Grid Search
# ===============================
for lr in param_grid["learning_rate"]:
    for depth in param_grid["max_depth"]:
        for n_est in param_grid["n_estimators"]:

            fold_r2 = []
            fold_rmse = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                model = LGBMRegressor(
                        learning_rate=lr,
                        n_estimators=n_est,
                        random_state=SEED,
                        early_stopping_rounds=100,
                        n_jobs=-1,
                        verbosity=-1
                    )

                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                )

                y_pred = model.predict(X_val)
                fold_r2.append(r2_score(y_val, y_pred))
                fold_rmse.append(root_mean_squared_error(y_val, y_pred))

            results.append({
                    "learning_rate": lr,
                    "max_depth": depth,
                    "n_estimators": n_est,
                    "RMSE_mean": np.mean(fold_rmse),
                    "RMSE_std": np.std(fold_rmse),
                    "R2_mean": np.mean(fold_r2)
                })

            print(
                    f"lr={lr:<5} depth={depth:<2} | "
                    f"n_est={n_est:<4} | RMSE={np.mean(fold_rmse):.3f}"
                )
            
#%% Grid search result
grid_result_df = pd.DataFrame(results)
grid_result_df = grid_result_df.sort_values("R2_mean", ascending=False)

best_row = grid_result_df.loc[grid_result_df["RMSE_mean"].idxmin()]
print(best_row)

#%% Find best parameters
best_params = {"learning_rate": best_row["learning_rate"], 
               "max_depth"        : int(best_row["max_depth"]),
               "n_estimators"   : int(best_row["n_estimators"])}

model = LGBMRegressor(
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
    **best_params)

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

#%% Save model / parameters
model.booster_.save_model(f'./hyperparameter_tuning/output/lightGBM_model_seed{SEED}.txt') 

with open(f'./hyperparameter_tuning/output/lightGBM_model_params_seed{SEED}.pkl', "wb") as f:
    pickle.dump(best_params, f)
    
#%% Reproduce model
import lightgbm as lgb

SEED = 23
with open(f'./hyperparameter_tuning/output/lightGBM_model_params_seed{SEED}.pkl', "rb") as f:
    loaded_best_params = pickle.load(f)
print(loaded_best_params)

# sklearn wrapper 대신 booster 직접 로드해서 predict
booster = lgb.Booster(model_file=f'./hyperparameter_tuning/output/lightGBM_model_seed{SEED}.txt')

y_test_pred  = booster.predict(X_test)
y_train_pred = booster.predict(X_train)

y_test_pred  = np.clip(y_test_pred, 0, 100)
y_train_pred = np.clip(y_train_pred, 0, 100)

train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test, y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse  = root_mean_squared_error(y_test, y_test_pred)
print(f"Train R2 score: {train_r2:.4f} - RMSE: {train_rmse:.4f}")
print(f"Test R2 score:  {test_r2:.4f} - RMSE: {test_rmse:.4f}")