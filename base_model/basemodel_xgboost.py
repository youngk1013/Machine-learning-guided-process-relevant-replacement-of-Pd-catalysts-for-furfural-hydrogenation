#%% Import libraries

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import random
from base_model.config import base_params
from utils import *
#%% Fix Seed

def set_seed(seed):    
    # Python 내장 random 모듈
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
SEED = 23 
set_seed(SEED)

#%% Load dataset / Preprocessing
dataset = pd.read_csv('./dataset/preprocessed_dataset_for_ML.csv')

#%% K-fold validation
set_seed(SEED)
params = base_params['XGBoost']
X_train, X_test, y_train, y_test = categorization(dataset, SEED)    
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

best_estimator = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = XGBRegressor(**params)

    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

    y_val_pred = model.predict(X_val)   
    y_val_pred = np.clip(y_val_pred, 0, 100)
    
    r2 = r2_score(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)
    best_estimator.append(model.best_iteration)
    
    print(f"Fold: {fold + 1} - R2 score: {r2:.4f} - RMSE: {rmse:.4f}")
    print(f"Best Estimator {model.best_iteration}")

#%% Re-train model

avg_estimator = int(np.mean(best_estimator))

model = XGBRegressor(
    n_estimators=avg_estimator,
    random_state=SEED,
    n_jobs=-1
)

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

#%% Save model
model.save_model(f"./base_model/output/xgb_model_seed{SEED}.json")

#%% Load model
loaded_model = XGBRegressor()
loaded_model.load_model(f"./base_model/output/xgb_model_seed{SEED}.json")

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
