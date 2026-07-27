#%% Import libraries

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
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
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": [3, 4, 6, 8, 10, 12],
    "n_estimators": [500, 1000, 2000, 3000, 5000]
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

            
                model = XGBRegressor(
                    learning_rate=lr,
                    max_depth=depth,
                    n_estimators=n_est,
                    random_state=SEED,
                    early_stopping_rounds=100,
                    n_jobs=-1,
                )

                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                y_pred = model.predict(X_val)
                fold_r2.append(r2_score(y_val, y_pred))
                fold_rmse.append(root_mean_squared_error(y_val, y_pred))

            results.append({
                "learning_rate": lr,
                "max_depth": depth,
                "n_estimators": n_est,
                "R2_mean": np.mean(fold_r2),
                "R2_std": np.std(fold_r2),
                "RMSE_mean": np.mean(fold_rmse),
                "RMSE_std": np.std(fold_rmse)
            })

            print(
                f"lr={lr:<5} depth={depth:<2} n_est={n_est:<4} | "
                f"R2={np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}"
            )

#%% Grid search result
grid_result_df = pd.DataFrame(results)
grid_result_df = grid_result_df.sort_values("R2_mean", ascending=False)

best_row = grid_result_df.loc[grid_result_df["RMSE_mean"].idxmin()]
print(best_row)

#%% Find best parameters
best_params = {"learning_rate": best_row["learning_rate"], 
               "max_depth": int(best_row["max_depth"]),
               "n_estimators": int(best_row["n_estimators"])}

model = XGBRegressor(
    random_state=SEED,
    n_jobs=-1, 
    **best_params)

model.fit(X_train, y_train, verbose=False)

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
    
model.save_model(f'./hyperparameter_tuning/output/xgb_model_seed_{SEED}.json')

with open(f'./hyperparameter_tuning/output/xgb_params_seed_{SEED}.pkl', "wb") as f:
    pickle.dump(best_params, f)
    
#%% Reproduce model
model = XGBRegressor()

model.load_model(f'./hyperparameter_tuning/output/xgb_model_seed_{SEED}.json')

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
