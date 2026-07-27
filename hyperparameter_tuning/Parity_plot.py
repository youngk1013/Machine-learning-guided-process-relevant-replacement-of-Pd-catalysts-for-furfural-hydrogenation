# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:16:36 2026

@author: USER
"""

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor 
from xgboost import XGBRegressor, DMatrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import seaborn as sns
import random
import re
import pickle

# from bayes_opt import BayesianOptimization

import os
os.chdir(r"\\165.132.128.132\PID_server\01. 개인폴더\김영근\[11] 촉매 AI")
#%% Fix Seed

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)

SEED = 23 
set_seed(SEED)

#%% Load dataset / Preprocessing

dataset = pd.read_csv('./dataset/preprocessed_dataset_for_ML.csv')
raw_data = dataset.copy()
print(list(dataset.columns))

before = dataset.shape[0]
dataset = dataset.drop_duplicates().reset_index(drop=True)
after = dataset.shape[0]

print(f"중복 제거 전 행 수 : {before}")
print(f"중복 제거 후 행 수 : {after}")
print(f"삭제된 중복 행 수 : {before-after}")
#%% After removal
am_start = dataset.columns.get_loc('Ca')
am_end  = dataset.columns.get_loc('Zn')

supp_start = dataset.columns.get_loc('Al2O3')
supp_end   = dataset.columns.get_loc('ZrO2')

pre_start = dataset.columns.get_loc('(NH4)2PdCl4')
pre_end = dataset.columns.get_loc('Zn(NO3)2')

prep_start = dataset.columns.get_loc('Unknown_preparation')
prep_end   = dataset.columns.get_loc('wet impregnation')

sol_start = dataset.columns.get_loc('1,2-dichloroethane')
sol_end   = dataset.columns.get_loc('water')

active_metal = dataset.columns[am_start:am_end + 1].tolist()
cat_support  = dataset.columns[supp_start:supp_end + 1].tolist()
preparation  = dataset.columns[prep_start:prep_end + 1].tolist()
precursor    = dataset.columns[pre_start:pre_end + 1].tolist()
solvent      = dataset.columns[sol_start:sol_end + 1].tolist()

print(f'Active metal : {len(active_metal)}개')
print(f'Support : {len(cat_support)}개')
print(f'Precursor : {len(precursor)}개')
print(f'Preparation : {len(preparation)}개')
print(f'Solvent : {len(solvent)}개')

print(dataset.shape)


#%% Create TF code

category_col = active_metal + cat_support + preparation + precursor + solvent
# category_col = active_metal + cat_support + preparation + solvent

dataset['TF_code'] = dataset[category_col].apply(lambda row: ''.join(['T' if v != 0 else 'F' for v in row]), axis=1)


dataset['Yield_bin'] = pd.cut(
    dataset['THFA_yield (%)'], 
    bins=[0, 20, 40, 60, 80, 100], 
    labels=['0-20', '20-40', '40-60', '60-80', '80-100'],
    include_lowest=True
)

# 2. TF_code + Yield_bin 조합으로 새로운 그룹 생성
dataset['TF_Yield_group'] = dataset['TF_code'] + '_' + dataset['Yield_bin'].astype(str)
group_counts = dataset['TF_Yield_group'].value_counts()

print(f"\nTotal unique TF_Yield_groups: {len(group_counts)}")
print(f"Groups with 1 sample: {(group_counts == 1).sum()}")
print(f"Groups with 2-5 samples: {((group_counts >= 2) & (group_counts <= 5)).sum()}")
print(f"Groups with 6+ samples: {(group_counts >= 6).sum()}")

#%% Categorization

train_val_indices = []
test_indices = []

# TF_Yield_group별로 직접 분할
for group in dataset['TF_Yield_group'].unique():
    group_mask = dataset['TF_Yield_group'] == group
    group_indices = dataset[group_mask].index.tolist()
    n_samples = len(group_indices)
    
    if n_samples <= 5:
        # 5개 이하는 모두 train_val set으로
        train_val_indices.extend(group_indices)
    else:
        # 6개 이상은 8:2 비율로 train_val / test 로 분할
        n_test = max(1, int(n_samples * 0.2)) # 최소 1개는 test 셋으로
        n_train_val = n_samples - n_test
        
        np.random.seed(SEED)
        shuffled_indices = np.random.permutation(group_indices)
        
        train_val_indices.extend(shuffled_indices[:n_train_val].tolist())
        test_indices.extend(shuffled_indices[n_train_val:].tolist())
        

#  최종 데이터셋 생성
df_train_val = dataset.loc[train_val_indices].copy()
df_test      = dataset.loc[test_indices].copy()

print("\n" + "="*60)
print("=== 데이터 분할 결과 ===")
print(f"Train_val samples: {len(df_train_val)} ({len(df_train_val)/len(dataset)*100:.1f}%)")
print(f"Test samples: {len(df_test)} ({len(df_test)/len(dataset)*100:.1f}%)")
print(f"Total: {len(df_train_val) + len(df_test)}")
print("="*60)

#%% Create final train / test set

X_train  = df_train_val.drop(columns=['THFA_yield (%)', 'TF_code', 'Yield_bin', 'TF_Yield_group'])
y_train  = df_train_val['THFA_yield (%)']

X_test = df_test.drop(columns=['THFA_yield (%)', 'TF_code', 'Yield_bin', 'TF_Yield_group'])
y_test = df_test['THFA_yield (%)']

#%%
SEED = 23

with open(f'./output/grid_search_tuned_model/catboost_params_seed_{SEED}', "rb") as f:
    loaded_best_params = pickle.load(f)

print(loaded_best_params)

model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=SEED,
    use_best_model=False,
    verbose=False,
    **loaded_best_params
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
#%%
with open(f'./output/grid_search_tuned_model/xgb_params_seed_{SEED}.pkl', "rb") as f:
    loaded_best_params = pickle.load(f)

print(loaded_best_params)

model = XGBRegressor(
    random_state=SEED,
    n_jobs=-1,
    **loaded_best_params
)

model.load_model(f'./output/grid_search_tuned_model/xgb_model_seed_{SEED}.json')

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
#%%
from lightgbm import LGBMRegressor
import lightgbm as lgb

SEED = 23
with open(f'./output/grid_search_tuned_model/lightGBM_model_params_seed{SEED}.pkl', "rb") as f:
    loaded_best_params = pickle.load(f)
print(loaded_best_params)

# sklearn wrapper 대신 booster 직접 로드해서 predict
booster = lgb.Booster(model_file=f'./output/grid_search_tuned_model/lightGBM_model_seed{SEED}.txt')

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
#%%
from sklearn.tree import DecisionTreeRegressor
with open(f'./output/grid_search_tuned_model/DT_model_params_seed{SEED}.pkl', "rb") as f:
    loaded_best_params = pickle.load(f)

print(loaded_best_params)

with open(f'./output/grid_search_tuned_model/DT_model_seed{SEED}.pkl', "rb") as f:
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
#%%
from sklearn.ensemble import RandomForestRegressor


with open(f'./output/grid_search_tuned_model/RF_model_params_seed{SEED}.pkl', "rb") as f:
    loaded_best_params = pickle.load(f)

print(loaded_best_params)

with open(f'./output/grid_search_tuned_model/RF_model_seed{SEED}.pkl', "rb") as f:
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
#%%
SEED = 23

with open(f'./output/grid_search_tuned_model/lr_model_seed_{SEED}.pkl',  'rb') as f:
    model = pickle.load(f)

with open(f'./output/grid_search_tuned_model/lr_scaler_seed_{SEED}.pkl', 'rb') as f:
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
#%% Ridge
with open(f'./output/grid_search_tuned_model/ridge_model_seed_{SEED}.pkl',  'rb') as f:
    model = pickle.load(f)

with open(f'./output/grid_search_tuned_model/ridge_scaler_seed_{SEED}.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'./output/grid_search_tuned_model/ridge_params_seed_{SEED}.pkl', 'rb') as f:
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
#%%SVR
X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)
with open(f'./output/grid_search_tuned_model/svr_model_seed_{SEED}.pkl',  'rb') as f:
    model = pickle.load(f)

with open(f'./output/grid_search_tuned_model/svr_scaler_seed_{SEED}.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'./output/grid_search_tuned_model/svr_params_seed_{SEED}.pkl', 'rb') as f:
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
#%% Lasso

with open(f'./output/grid_search_tuned_model/lasso_model_seed_{SEED}.pkl',  'rb') as f:
    model = pickle.load(f)

with open(f'./output/grid_search_tuned_model/lasso_scaler_seed_{SEED}.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'./output/grid_search_tuned_model/lasso_params_seed_{SEED}.pkl', 'rb') as f:
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



#%% Parity Plot 

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

scatter = ax.scatter(
    y_test, y_test_pred,
    c='#56B4E9',
    edgecolors='white',
    linewidths=0.5,
    s=60,
    alpha=0.85,
    zorder=3,
    label='Test set'
)

lims = [-5, 105]   # 넉넉하게
ax.plot(lims, lims,
        color='black', linewidth=1.5,
        linestyle='--', zorder=2)

ax.fill_between(lims,
                [v - 10 for v in lims],
                [v + 10 for v in lims],
                color='gray', alpha=0.12, zorder=1, label='±10%')

textstr = (f'$R^2$ = {test_r2:.4f}\n'
           f'RMSE = {test_rmse:.2f}')
ax.text(0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='white',
                  edgecolor='#CCCCCC',
                  alpha=0.9))

ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.set_xlabel('Actual THFA Yield (%)', fontsize=16)
ax.set_ylabel('Predicted THFA Yield (%)', fontsize=16)


# ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.set_aspect('equal')

# ── Spine 정리 ───────────────────────────────────────────
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

ax.tick_params(axis='both', which='major', labelsize=16, direction='in', length=4)

plt.tight_layout()
# plt.savefig('./output/parity_plot_test.png', dpi=300, bbox_inches='tight')
plt.show()





