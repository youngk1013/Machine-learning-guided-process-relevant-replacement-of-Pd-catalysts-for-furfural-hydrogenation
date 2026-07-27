# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:42:24 2026

@author: USER
"""

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
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

#%% Grid Search (alpha 튜닝)

alpha_grid = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

results = []

for alpha in alpha_grid:
    fold_r2 = []
    fold_rmse = []
    fold_zero_coef = []

    for tr_idx, val_idx in kf.split(X_train):
            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]
    
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_val_scaled = fold_scaler.transform(X_val)
    
            fold_model = Lasso(
                alpha=alpha,
                max_iter=10000,
                random_state=SEED,
            )
            fold_model.fit(X_tr_scaled, y_tr)
            y_pred = fold_model.predict(X_val_scaled)
    
            fold_r2.append(r2_score(y_val, y_pred))
            fold_rmse.append(root_mean_squared_error(y_val, y_pred))
            fold_zero_coef.append(np.sum(fold_model.coef_ == 0))
    
    results.append({
        'alpha'           : alpha,
        'R2_mean'         : np.mean(fold_r2),
        'R2_std'          : np.std(fold_r2),
        'RMSE_mean'       : np.mean(fold_rmse),
        'RMSE_std'        : np.std(fold_rmse),
        'n_zero_coef_mean': np.mean(fold_zero_coef),
    })
    
    print(
        f"alpha={alpha:<8} | "
        f"R2={np.mean(fold_r2):.4f} +/- {np.std(fold_r2):.4f}  "
        f"RMSE={np.mean(fold_rmse):.4f} +/- {np.std(fold_rmse):.4f}  "
        f"zero_coef={np.mean(fold_zero_coef):.1f}"
    )

#%% Grid search result

grid_result_df = pd.DataFrame(results).sort_values('RMSE_mean', ascending=True)
print("\n" + "="*60)
print(grid_result_df.to_string(index=False))

best_row    = grid_result_df.iloc[0]
best_alpha  = best_row['alpha']
best_params = {'alpha': best_alpha}

print(f"\n[Best] alpha={best_alpha} CV R2={best_row['R2_mean']:.4f} CV RMSE={best_row['RMSE_mean']:.4f}")

#%% Train final model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Lasso(alpha=best_alpha, max_iter=10000, random_state=SEED)
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

#%% Save model / scaler / params

with open(f'./hyperparameter_tuning/output/lasso_model_seed_{SEED}.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(f'./hyperparameter_tuning/output/lasso_scaler_seed_{SEED}.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(f'./hyperparameter_tuning/output/lasso_params_seed_{SEED}.pkl', 'wb') as f:
    pickle.dump(best_params, f)

#%% Reproduce model

SEED = 23

with open(f'./hyperparameter_tuning/output/lasso_model_seed_{SEED}.pkl',  'rb') as f:
    model = pickle.load(f)

with open(f'./hyperparameter_tuning/output/lasso_scaler_seed_{SEED}.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'./hyperparameter_tuning/output/lasso_params_seed_{SEED}.pkl', 'rb') as f:
    loaded_best_params = pickle.load(f)

print(loaded_best_params)

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
