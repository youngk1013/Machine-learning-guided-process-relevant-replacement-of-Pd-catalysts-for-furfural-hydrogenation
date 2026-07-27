# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:03:16 2026

@author: USER
"""

SEED = 23

base_params = {
    'CatBoost': {'iterations' : 3000, 'loss_function' : 'RMSE', 'eval_metric' : 'RMSE', 'random_seed' : SEED, 'verbose': 0},
    'XGBoost' : {'n_estimators' : 3000, 'early_stopping_rounds' : 100,  'random_state' : SEED, 'n_jobs' : -1},
    'DecisionTree' : {'random_state' : SEED},
    'RandomForest': {'random_state' : SEED, 'n_jobs' : -1},
    'LightGBM': {'random_state' : SEED, 'n_jobs' : -1, 'n_estimators' : 3000, 'early_stopping_rounds' : 100, 'verbosity' : -1},
    'Lasso': {'max_iter' : 10000, 'random_state' : SEED},
    'Linear': {'n_jobs' : -1}}