# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:16:36 2026

@author: USER
"""

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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


#%% Load dataset / Preprocessing
dataset = pd.read_csv('./dataset/preprocessed_dataset_for_ML.csv')

#%% Perform grid search
X_train, X_test, y_train, y_test = categorization(dataset, SEED)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

param_grid = {
    "n_estimators": [100, 300, 500, 1000], 
    "max_depth": [None, 4,  6, 8, 10, 15],
    "min_samples_leaf": [1, 3, 5]
}

results = []

# ===============================
# Grid Search
# ===============================
for n_est in param_grid["n_estimators"]:
    for depth in param_grid["max_depth"]:
        for leaf in param_grid["min_samples_leaf"]:

            fold_r2 = []
            fold_rmse = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=SEED,
                    n_jobs=-1
                )

                model.fit(X_tr, np.ravel(y_tr))

                y_pred = model.predict(X_val)
                fold_r2.append(r2_score(y_val, y_pred))
                fold_rmse.append(root_mean_squared_error(y_val, y_pred))

            results.append({
                "n_estimators": n_est,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "RMSE_mean": np.mean(fold_rmse),
                "RMSE_std": np.std(fold_rmse),
                "R2_mean": np.mean(fold_r2)
            })

            print(
                f"n_est={n_est:<4} depth={str(depth):<4} leaf={leaf:<2} | "
                f"RMSE={np.mean(fold_rmse):.3f}"
            )
            
#%% Grid search result

grid_result_df = pd.DataFrame(results)
grid_result_df = grid_result_df.sort_values("R2_mean", ascending=False)

best_row = grid_result_df.loc[grid_result_df["RMSE_mean"].idxmin()]
print(best_row)

#%% Find best parameters

best_params = {"n_estimators"       : int(best_row["n_estimators"]), 
               "max_depth"          : best_row["max_depth"],
               "min_samples_leaf"   : int(best_row["min_samples_leaf"])}

if pd.isna(best_params["max_depth"]):
    best_params["max_depth"] = None

model = RandomForestRegressor(
                    random_state=SEED,
                    n_jobs=-1,
                    **best_params
                )

model.fit(X_train, np.ravel(y_train))

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
with open(f'./hyperparameter_tuning/output/RF_model_seed{SEED}.pkl', "wb") as f:
    pickle.dump(model, f)

with open(f'./hyperparameter_tuning/output/RF_model_params_seed{SEED}.pkl', "wb") as f:
    pickle.dump(best_params, f)

#%% Reproduce model
SEED = 23

with open(f'./hyperparameter_tuning/output/RF_model_params_seed{SEED}.pkl', "rb") as f:
    loaded_best_params = pickle.load(f)

print(loaded_best_params)

with open(f'./hyperparameter_tuning/output/RF_model_seed{SEED}.pkl', "rb") as f:
    model = pickle.load(f)

y_test_pred  = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred  = np.clip(y_test_pred, 0, 100)
y_train_pred = np.clip(y_train_pred, 0, 100)

train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test, y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse  = root_mean_squared_error(y_test, y_test_pred)

print(f"Train R2 score: {train_r2:.4f} - RMSE: {train_rmse:.4f}")
print(f"Test R2 score:  {test_r2:.4f} - RMSE: {test_rmse:.4f}")