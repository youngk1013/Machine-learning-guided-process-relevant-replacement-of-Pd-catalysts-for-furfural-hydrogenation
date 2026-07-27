#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import seaborn as sns
import random
import re
import pickle

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

#%% Dataset analysis
#1. Active metal
am_start = dataset.columns.get_loc('Ca')
am_end  = dataset.columns.get_loc('Zn')
active_metal = dataset.columns[am_start:am_end + 1].tolist()

#2. Support composition
supp_start = dataset.columns.get_loc('Al2O3')
supp_end   = dataset.columns.get_loc('ZrO2')
cat_support  = dataset.columns[supp_start:supp_end + 1].tolist()

#3.Metal precursor 
pre_start = dataset.columns.get_loc('(NH4)2PdCl4')
pre_end = dataset.columns.get_loc('Zn(NO3)2')
precursor    = dataset.columns[pre_start:pre_end + 1].tolist()

#4. Preparation method
prep_start = dataset.columns.get_loc('Unknown_preparation')
prep_end   = dataset.columns.get_loc('wet impregnation')
preparation  = dataset.columns[prep_start:prep_end + 1].tolist()

#5. Solvent
sol_start = dataset.columns.get_loc('1,2-dichloroethane')
sol_end   = dataset.columns.get_loc('water')
solvent   = dataset.columns[sol_start:sol_end + 1].tolist()

#6. Float columns
float_cols= ['Reduction_temp', 'Reduction_time', 'Calcination_temp', 'Calcination_time', 
             'Furfural (mg)', 'Catalyst amount (mg)', 'Operating_temp', 'Operating_pressure', 'Operating_time',
             'Stirring rate (rpm)', 'Substrate to metal ratio (mmol/mmol)', 'Substrate concentration (mg/ml)', 'THFA_yield (%)']

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
#%% K-fold validation

set_seed(SEED)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

best_iterations = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = DecisionTreeRegressor(random_state=SEED)
    
    model.fit(X_tr, np.array(y_tr).ravel())

    y_val_pred = model.predict(X_val)   
    y_val_pred = np.clip(y_val_pred, 0, 100)
    
    r2 = r2_score(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)
    fold_models.append(model)
    print(f"Fold: {fold + 1} - R2 score: {r2:.4f} - RMSE: {rmse:.4f}")

#%% Re-train model

model = DecisionTreeRegressor(random_state=SEED)

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

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi = 300)

axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=25, label='_nolegend_')

min_val_train = min(min(np.array(y_train)), y_train_pred.min())
max_val_train = max(max(np.array(y_train)), y_train_pred.max()) 

axes[0].plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Yield (%)', fontsize=12)
axes[0].set_ylabel('Predicted Yield (%)', fontsize=12)
axes[0].set_title(f'Train Set (R² = {train_r2:.4f})', fontsize=14, fontweight='bold') 
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- 기존과 동일: Test set Plot ---
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=25, color='orange', label='_nolegend_')
min_val_test = min(min(np.array(y_test)), y_test_pred.min())
max_val_test = max(max(np.array(y_test)), y_test_pred.max()) 
axes[1].plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Yield (%)', fontsize=12)
axes[1].set_ylabel('Predicted Yield (%)', fontsize=12)
axes[1].set_title(f'Test Set (R² = {test_r2:.4f})', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Save model
with open(f"./output/base_model/DT_model_seed{SEED}.pkl", "wb") as f:
    pickle.dump(model, f)

#%% Load model
with open(f"./output/base_model/DT_model_seed{SEED}.pkl", "rb") as f:
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