#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import random
import pickle
from base_model.config import base_params
from utils import *
#%% Fix Seed

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
SEED = 23 
set_seed(SEED)

#%% Load dataset / Preprocessing
dataset = pd.read_csv('./dataset/preprocessed_dataset_for_ML.csv')

#%% Preprocessing
X_train, X_test, y_train, y_test = categorization(dataset, SEED)

X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#%% K-fold validation
set_seed(SEED)
params = base_params['Linear']  
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

best_iterations = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = LinearRegression(**params)
    model.fit(X_tr, np.array(y_tr).ravel())

    y_val_pred = model.predict(X_val)   
    y_val_pred = np.clip(y_val_pred, 0, 100)
    
    r2 = r2_score(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)
    fold_models.append(model)
    print(f"Fold: {fold + 1} - R2 score: {r2:.4f} - RMSE: {rmse:.4f}")

#%% Re-train model
model = LinearRegression()
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
with open(f"./base_model/output/Linear_model_seed{SEED}.pkl", "wb") as f:
    pickle.dump(model, f)

#%% Load model
with open(f"./base_model/output/Linear_model_seed{SEED}.pkl", "rb") as f:
    loaded_model = pickle.load(f)

y_test_pred = loaded_model.predict(X_test)   
y_train_pred = loaded_model.predict(X_train)

y_test_pred = np.clip(y_test_pred, 0, 100)
y_train_pred = np.clip(y_train_pred, 0, 100)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
print(f"Train R2 score: {train_r2:.4f} - RMSE: {train_rmse:.4f}")
print(f"Test R2 score:  {test_r2:.4f} - RMSE: {test_rmse:.4f}")