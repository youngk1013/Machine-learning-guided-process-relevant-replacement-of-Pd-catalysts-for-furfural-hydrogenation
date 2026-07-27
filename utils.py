# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:06:19 2026

@author: USER
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#%%
def categorization(dataset, SEED):
    #1. active metal
    am_start = dataset.columns.get_loc('Ca')
    am_end  = dataset.columns.get_loc('Zn')
    active_metal = dataset.columns[am_start:am_end + 1].tolist()

    #2. Catalyst support
    supp_start = dataset.columns.get_loc('Al2O3')
    supp_end   = dataset.columns.get_loc('ZrO2')
    cat_support  = dataset.columns[supp_start:supp_end + 1].tolist()

    #3. Metal precursor 
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
    
    category_col = active_metal + cat_support + preparation + precursor + solvent
    dataset['TF_code'] = dataset[category_col].apply(lambda row: ''.join(['T' if v != 0 else 'F' for v in row]), axis=1)

    dataset['Yield_bin'] = pd.cut(
        dataset['THFA_yield (%)'], 
        bins=[0, 20, 40, 60, 80, 100], 
        labels=['0-20', '20-40', '40-60', '60-80', '80-100'],
        include_lowest=True
    )

    dataset['TF_Yield_group'] = dataset['TF_code'] + '_' + dataset['Yield_bin'].astype(str)

    train_val_indices = []
    test_indices = []

    for group in dataset['TF_Yield_group'].unique():
        group_mask = dataset['TF_Yield_group'] == group
        group_indices = dataset[group_mask].index.tolist()
        n_samples = len(group_indices)
        
        if n_samples <= 5:
            train_val_indices.extend(group_indices)
        else:
            n_test = max(1, int(n_samples * 0.2)) 
            n_train_val = n_samples - n_test
            
            np.random.seed(SEED)
            shuffled_indices = np.random.permutation(group_indices)
            
            train_val_indices.extend(shuffled_indices[:n_train_val].tolist())
            test_indices.extend(shuffled_indices[n_train_val:].tolist())

    df_train_val = dataset.loc[train_val_indices].copy()
    df_test      = dataset.loc[test_indices].copy()
    
    X_train  = df_train_val.drop(columns=['THFA_yield (%)', 'TF_code', 'Yield_bin', 'TF_Yield_group'])
    y_train  = df_train_val['THFA_yield (%)']

    X_test = df_test.drop(columns=['THFA_yield (%)', 'TF_code', 'Yield_bin', 'TF_Yield_group'])
    y_test = df_test['THFA_yield (%)']
    
    return X_train, X_test, y_train, y_test

def show_parity_plot(y_train, y_train_pred, y_test, y_test_pred):
    #Calculate R2 score
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    #Show parity plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi = 300)

    axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=25, label='_nolegend_')

    min_val_train = min(min(np.array(y_train)), y_train_pred.min())
    max_val_train = max(max(np.array(y_train)), y_train_pred.max()) 
    
    #Parity plot for train set
    axes[0].plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Yield (%)', fontsize=12)
    axes[0].set_ylabel('Predicted Yield (%)', fontsize=12)
    axes[0].set_title(f'Train Set (R² = {train_r2:.4f})', fontsize=14, fontweight='bold') 
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #Parity plot for test set
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
    
#%% Save dataset

# X_train.to_csv('./dataset/ML_dataset_final_x_train.csv', index=False)
# y_train.to_csv('./dataset/ML_dataset_final_y_train.csv', index=False)
# X_test.to_csv('./dataset/ML_dataset_final_x_test.csv', index=False)
# y_test.to_csv('./dataset/ML_dataset_final_y_test.csv', index=False)