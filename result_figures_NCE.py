# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:05:44 2026

@author: USER
"""

#%% Import libraries
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import random
import shap
import pickle
import joblib
import string
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib import cm, rcParams
from matplotlib.lines import Line2D

from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import InconsistentVersionWarning

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor 

import warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

#%% Load pre-trained ML model

def set_seed(seed):    
    random.seed(seed) 
    np.random.seed(seed)

SEED = 23 
set_seed(SEED)

X_train = pd.read_csv('./dataset/ML_dataset_final_x_train.csv')
y_train = pd.read_csv('./dataset/ML_dataset_final_y_train.csv')
X_test  = pd.read_csv('./dataset/ML_dataset_final_x_test.csv')
y_test  = pd.read_csv('./dataset/ML_dataset_final_y_test.csv')

model = XGBRegressor(random_state=SEED, n_jobs=-1)
model.load_model(f'./hyperparameter_tuning/output/xgb_model_seed_{SEED}.json')

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred = np.clip(y_test_pred, 0, 100)
y_train_pred = np.clip(y_train_pred, 0, 100)

XGB_train_r2 = r2_score(y_train, y_train_pred)
XGB_test_r2 = r2_score(y_test, y_test_pred)

XGB_train_rmse = root_mean_squared_error(y_train, y_train_pred)
XGB_test_rmse = root_mean_squared_error(y_test, y_test_pred)

print(f"Train R2 score: {XGB_train_r2:.4f} - RMSE: {XGB_train_rmse:.4f}")
print(f"Test R2 score:  {XGB_test_r2:.4f} - RMSE: {XGB_test_rmse:.4f}")

#%% Fig. 2 ML predictive basis / SHAP / global PDP

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 245 * MM_TO_INCH

FS_PANEL = 8.0
FS_LABEL = 7.0
FS_TICK = 6.0
FS_TEXT = 6.0
FS_SMALL = 5.5
FS_CBAR = 6.0
PANEL_PAD = 5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TEXT,

    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_PANEL,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 0.9,
    'lines.markersize': 3.5,

    'legend.fontsize': FS_SMALL,
    'legend.frameon': False,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})


def draw_2d_pdv(
    fig,
    subspec,
    model,
    X_train,
    prop_1,
    prop_2,
    xlim=None,
    ylim=None,
    x_ticks=None,
    y_ticks=None,
    x_label='',
    y_label='',
    title='',
    grid_resolution=30,
    cmap='RdYlBu_r',
    zmin=0,
    zmax=100
):

    X_pdp = X_train.astype(np.float64)
    X_custom = X_pdp.copy()

    if prop_1 == 'Calcination_temp':
        X_custom = X_custom[X_custom[prop_1] >= 100]

    if prop_2 == 'Calcination_time':
        X_custom = X_custom[X_custom[prop_2] <= 10]

    if prop_1 == 'Reduction_temp':
        X_custom = X_custom[X_custom[prop_1] >= 100]

    if prop_2 == 'Reduction_time':
        X_custom = X_custom[X_custom[prop_2] <= 10]

    if prop_1 == 'Operating_temp':
        X_custom = X_custom[X_custom[prop_1] >= 50]

    if prop_2 == 'Operating_time':
        X_custom = X_custom[X_custom[prop_2] <= 10]

    X_custom = X_custom.copy()

    X_custom[prop_1] = pd.to_numeric(X_custom[prop_1], errors='coerce')

    X_custom[prop_2] = pd.to_numeric(X_custom[prop_2], errors='coerce')

    X_custom = X_custom.dropna(subset=[prop_1, prop_2])

    custom_grid = {
        prop_1: np.linspace(
            X_custom[prop_1].min(),
            X_custom[prop_1].max(),
            grid_resolution
        ),
        prop_2: np.linspace(
            X_custom[prop_2].min(),
            X_custom[prop_2].max(),
            grid_resolution
        )
    }

    if (prop_1 == 'Operating_temp' and prop_2 == 'Operating_time'):
        custom_grid = {
            prop_1: np.linspace(
                50,
                X_custom[prop_1].max(),
                grid_resolution
            ),
            prop_2: np.linspace(
                X_custom[prop_2].min(),
                10,
                grid_resolution
            )
        }

    if (prop_1 == 'Reduction_temp' and prop_2 == 'Reduction_time'):
        custom_grid = {
            prop_1: np.linspace(
                100,
                X_custom[prop_1].max(),
                grid_resolution
            ),
            prop_2: np.linspace(
                X_custom[prop_2].min(),
                10,
                grid_resolution
            )
        }

    if (prop_1 == 'Calcination_temp' and prop_2 == 'Calcination_time'):
        custom_grid = {
            prop_1: np.linspace(
                100,
                X_custom[prop_1].max(),
                grid_resolution
            ),
            prop_2: np.linspace(
                X_custom[prop_2].min(),
                10,
                grid_resolution
            )
        }

    #PDV calculation
    pd_results = partial_dependence(
        model,
        X_pdp,
        features=[prop_1, prop_2],
        custom_values=custom_grid
    )

    x_axis = np.asarray(
        pd_results['grid_values'][0],
        dtype=float
    )

    y_axis = np.asarray(
        pd_results['grid_values'][1],
        dtype=float
    )

    Xg, Yg = np.meshgrid(
        x_axis,
        y_axis,
        indexing='ij'
    )

    Z_raw = np.asarray(
        pd_results['average'][0],
        dtype=float
    ).reshape(
        len(x_axis),
        len(y_axis)
    )

    #Colorbar setting
    clip_zmin = (-np.inf if zmin is None else zmin)
    clip_zmax = (np.inf if zmax is None else zmax)

    Z = np.clip(Z_raw, clip_zmin, clip_zmax)
    
    #set axes
    gs = subspec.subgridspec(2, 3,
        width_ratios=[1.0, 5.0, 0.24], height_ratios=[1.0, 5.0],
        wspace=0.06, hspace=0.06)

    ax_joint = fig.add_subplot(gs[1, 1])
    ax_marg_x = fig.add_subplot(gs[0, 1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 0], sharey=ax_joint)

    ax_cb = fig.add_subplot(gs[1, 2])

    cp = ax_joint.contourf(Xg, Yg, Z, levels=40, cmap=cmap, antialiased=False)

    for collection in getattr(
        cp,
        'collections',
        []
    ):
        collection.set_edgecolor('face')
        collection.set_linewidth(0.0)
        collection.set_antialiased(False)
        collection.set_rasterized(True)

    if xlim is not None:
        ax_joint.set_xlim(*xlim)
        ax_marg_x.set_xlim(*xlim)

    if ylim is not None:
        ax_joint.set_ylim(*ylim)
        ax_marg_y.set_ylim(*ylim)

    sns.histplot(
        data=X_custom,
        x=prop_1,
        ax=ax_marg_x,
        color='gray',
        kde=True,
        alpha=0.28,
        bins=20,
        element='step',
        linewidth=0.7
    )

    ax_marg_x.set_xlabel('')

    ax_marg_x.set_ylabel('Density', fontsize=FS_SMALL, labelpad=1)

    ax_marg_x.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False
    )

    ax_marg_x.set_yticks([])

    sns.histplot(
        data=X_custom,
        y=prop_2,
        ax=ax_marg_y,
        color='gray',
        kde=True,
        alpha=0.28,
        bins=15,
        element='step',
        linewidth=0.7
    )

    ax_marg_y.invert_xaxis()

    ax_marg_y.set_xlabel('Density', fontsize=FS_SMALL, labelpad=1)

    ax_marg_y.set_ylabel(y_label, fontsize=FS_LABEL, labelpad=7)

    ax_marg_y.yaxis.set_label_position('left')

    ax_marg_y.yaxis.tick_left()
    ax_marg_y.set_xticks([])


    if x_ticks is not None:
        ax_joint.set_xticks(x_ticks)

    if y_ticks is not None:
        ax_marg_y.set_yticks(y_ticks)

    ax_joint.set_xlabel(x_label, fontsize=FS_LABEL)
    ax_joint.set_ylabel('')

    ax_joint.tick_params(
        axis='x',
        labelsize=FS_TICK,
        width=0.6,
        length=2.5,
        direction='out'
    )

    ax_joint.tick_params(
        axis='y',
        which='both',
        left=False,
        labelleft=False
    )

    ax_marg_y.tick_params(
        axis='y',
        labelsize=FS_TICK,
        width=0.6,
        length=2.5,
        direction='out'
    )

    ax_joint.set_aspect(
        'auto'
    )

    for current_ax in [
        ax_joint,
        ax_marg_x,
        ax_marg_y
    ]:
        for spine in current_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

        current_ax.grid(False)

    cbar = fig.colorbar(cp, cax=ax_cb)
    cbar.set_label('PDV', fontsize=FS_CBAR, labelpad=3)
    cbar.ax.tick_params(labelsize=FS_SMALL, width=0.6, length=2, direction='out')
    cbar.outline.set_linewidth(0.6)

    return {
    'joint': ax_joint,
    'marg_x': ax_marg_x,
    'marg_y': ax_marg_y,
    'colorbar_ax': ax_cb,
    'contour': cp
    }

def draw_1d_pdv(ax, model, X_train, feature,  xlabel, title, color, xlim=None, x_ticks=None):  
    X_pdp = X_train.astype(np.float64)
    
    display = PartialDependenceDisplay.from_estimator(
        model, X_pdp, features=[feature], kind='average',
        centered=False, grid_resolution=50,  ax=ax,
        line_kw={'color': color, 'linewidth': 1.2})

    real_ax = display.axes_[0, 0]

    if xlim is not None:
        real_ax.set_xlim(*xlim)
    if x_ticks is not None:
        real_ax.set_xticks(x_ticks)

    real_ax.tick_params(
        axis='both',
        which='major',
        labelsize=FS_TICK,
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    real_ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=3)

    real_ax.set_ylabel('PDV', fontsize=FS_LABEL, labelpad=3)

    real_ax.grid(False)
    real_ax.xaxis.grid(False)
    real_ax.yaxis.grid(False)

    for spine in real_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    return real_ax

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

outer = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.2, 1.0, 1.0], wspace=0.28, hspace=0.20)

#Parity plot
ax1 = fig.add_subplot(outer[0, 0])
scatter = ax1.scatter(
    y_test,
    y_test_pred,
    c='#56B4E9',
    edgecolors='white',
    linewidths=0.45,
    s=22,
    alpha=0.85,
    zorder=3,
    label='Test set'
)

lims = [-5, 105]

ax1.plot(lims, lims, color='black', linewidth=0.9, linestyle='--', zorder=2)

ax1.fill_between(
    lims,
    [
        value - 10
        for value in lims
    ],
    [
        value + 10
        for value in lims
    ],
    color='gray',
    alpha=0.12,
    zorder=1,
    label='±10%'
)

textstr = (
    f'$R^2$ = {XGB_test_r2:.4f}\n'
    f'RMSE = {XGB_test_rmse:.2f}'
)

ax1.text(
    0.05,
    0.95,
    textstr,
    transform=ax1.transAxes,
    fontsize=FS_TEXT,
    verticalalignment='top',
    bbox={
        'boxstyle': 'round,pad=0.3',
        'facecolor': 'white',
        'edgecolor': '#CCCCCC',
        'linewidth': 0.6,
        'alpha': 0.9
    }
)

ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.set_xlabel('Actual THFA yield (%)', fontsize=FS_LABEL)
ax1.set_ylabel('Predicted THFA yield (%)', fontsize=FS_LABEL)
ax1.set_aspect('equal', adjustable='box')
ax1.set_anchor('C')

for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

ax1.spines['left'].set_linewidth(0.6)
ax1.spines['bottom'].set_linewidth(0.6)

ax1.tick_params(axis='both', which='major', labelsize=FS_TICK, direction='out', length=2.5, width=0.6)


#SHAP summary plot
shap_grid = outer[0, 1].subgridspec(1, 2, width_ratios=[0.16, 1.0], wspace=0.0)
ax2_margin = fig.add_subplot(shap_grid[0, 0])
ax2_margin.axis('off')
ax2 = fig.add_subplot(shap_grid[0, 1])
axes_before_shap = set(fig.axes)
plt.sca(ax2)

rename_dict = {
    'Operating_temp': 'Operating temperature',
    'Operating_time': 'Operating time',
    'Operating_pressure': 'Operating pressure',
    'Furfural (mg)': 'Furfural amount',
    'Active metal_Ni': 'Ni',
    'Substrate concentration (mg/ml)': 'Substrate concentration',
    'Substrate to metal ratio (mmol/mmol)': 'Substrate-to-metal ratio',
    'Stirring rate (rpm)': 'Stirring rate',
    'Reduction_temp': 'Reduction temperature',
    'Reduction_time': 'Reduction time',
    'Calcination_temp': 'Calcination temperature',
    'Calcination_time': 'Calcination time',
    'water': 'Water',
    'ethanol': 'Ethanol'}

X_train_renamed = X_train.rename(columns=rename_dict)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_renamed)
mean_abs_shap = (np.abs(shap_values).mean(axis=0))

shap_df = pd.DataFrame({'feature': X_train_renamed.columns,
                        'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)

shap.summary_plot(shap_values, X_train_renamed, max_display=15, show=False, plot_size=None)

for line in list(ax2.lines):

    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)

    is_horizontal_line = (ydata.size >= 2 and np.allclose(ydata, ydata[0]) and not np.allclose(xdata, xdata[0]))

    if is_horizontal_line:
        line.remove()

ax2.grid(False)
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)

ax2.set_xlabel('SHAP value (impact on model output)', fontsize=FS_LABEL)
ax2.tick_params(axis='x', labelsize=FS_TICK, width=0.6, length=2.5, direction='out')
ax2.tick_params(axis='y', labelsize=FS_SMALL, width=0.6, length=0, pad=2)

for label in ax2.get_yticklabels():
    label.set_fontsize(FS_SMALL)
    label.set_fontweight('normal')
    label.set_horizontalalignment('right')

fig.canvas.draw()
new_shap_axes = [
    current_ax
    for current_ax in fig.axes
    if (current_ax not in axes_before_shap and current_ax is not ax2)]

if new_shap_axes:
    shap_cbar_ax = new_shap_axes[-1]

    shap_cbar_ax.set_ylabel('Feature value', fontsize=FS_LABEL, labelpad=4)
    shap_cbar_ax.tick_params(labelsize=FS_SMALL, width=0.6, length=2)

#2D PDP plot
panel_c = draw_2d_pdv(
    fig,
    outer[1, 0],
    model,
    X_train,
    prop_1='Calcination_temp',
    prop_2='Calcination_time',
    xlim=(100, 800),
    ylim=(1, 8),
    x_ticks=np.arange(100, 801, 100),
    y_ticks=np.arange(1, 9, 1),
    x_label='Calcination temperature (°C)',
    y_label='Calcination time (h)',
    title='',
    cmap='RdYlBu_r'
)

panel_d = draw_2d_pdv(
    fig,
    outer[1, 1],
    model,
    X_train,
    prop_1='Reduction_temp',
    prop_2='Reduction_time',
    xlim=(100, 800),
    ylim=(1, 8),
    x_ticks=np.arange(100, 801, 100),
    y_ticks=np.arange(1, 9, 1),
    x_label='Reduction temperature (°C)',
    y_label='Reduction time (h)',
    title='',
    cmap='RdYlBu_r'
)

panel_e = draw_2d_pdv(
    fig,
    outer[2, 0],
    model,
    X_train,
    prop_1='Operating_temp',
    prop_2='Operating_time',
    xlim=(50, 250),
    ylim=(1, 10),
    x_ticks=np.arange(50, 251, 50),
    y_ticks=np.arange(1, 11, 1),
    x_label='Operating temperature (°C)',
    y_label='Operating time (h)',
    title='',
    cmap='RdYlBu_r'
)

#1D PDP plot
inner = outer[2, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.34)
ax_f = fig.add_subplot(inner[0, 0])
ax_g = fig.add_subplot(inner[1, 0])

ax_f = draw_1d_pdv(
    ax_f,
    model,
    X_train,
    feature='Operating_pressure',
    xlabel='Operating pressure (bar)',
    title='',
    color='tab:blue',
    xlim=(0, 40),
    x_ticks=np.arange(0, 41, 10)
)

ax_g = draw_1d_pdv(
    ax_g,
    model,
    X_train,
    feature='Stirring rate (rpm)',
    xlabel='Stirring rate (rpm)',
    title='',
    color='tab:olive',
    xlim=(200, 1000),
    x_ticks=np.arange(200, 1001, 200)
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.045,
    top=0.985
)

fig.canvas.draw()
#Position setting
left_panel_x = ax1.get_position().x0
right_panel_x = ax_f.get_position().x0

panel_label_pad = 0.006

panel_label_positions = [
    ('a', left_panel_x, ax1.get_position().y1 + panel_label_pad),
    ('b', right_panel_x, ax2.get_position().y1 + panel_label_pad),
    ('c', left_panel_x, panel_c['marg_x'].get_position().y1 + panel_label_pad),
    ('d', right_panel_x, panel_d['marg_x'].get_position().y1 + panel_label_pad),
    ('e', left_panel_x, panel_e['marg_x'].get_position().y1 + panel_label_pad),
    ('f', right_panel_x, ax_f.get_position().y1 + panel_label_pad),
    ('g', right_panel_x, ax_g.get_position().y1 + panel_label_pad)]

for panel_label, panel_x, panel_y in panel_label_positions:
    fig.text(panel_x, panel_y, panel_label, fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom')
    
# plt.savefig('./figure_2.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_2.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Fig. 3 Pd-relative operating-window

X_train_float = X_train.astype(float)
noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']

Pd_only_df = X_train_float[(X_train_float['Pd'] > 0) & ((X_train_float[active_metal] > 0).sum(axis=1) == 1)]

condition_1 = X_train_float['Ni'] > 0
condition_2 = X_train_float[noble_metal].sum(axis=1) == 0
Ni_only_df = X_train_float[condition_1 & condition_2]

X_train_float = X_train.astype(float)

noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']
noble_data = X_train_float[X_train_float[noble_metal].sum(axis=1) > 0]
non_noble = X_train_float[(X_train_float[noble_metal].sum(axis=1) == 0)]

features = ['Operating_temp', 'Operating_pressure', 'Operating_time']
x_labels = ['Operating temperature (°C)', ' Operating pressure (bar)', 'Operating time (h)']
x_lims = [(50, 200), (0, 40), (0, 10)]
x_ticks = [
    (50, 100, 150, 200),
    (0, 10, 20, 30, 40),
    (0, 2, 4, 6, 8, 10)
]
y_lims = [[(-12, 62), (-12, 62), (-12, 62)]]

temp_range = np.arange(120, 201, 1)
pres_range = np.arange(10, 40.5, 0.5)
selected_support = 'Al2O3' 
selected_solvent = '2-propanol' 
solvent_amount   = 40
selected_preparation = 'wet impregnation'
calc_temp, calc_time = 500, 6
reduc_temp, reduc_time = 400, 4
oper_time =  6
stir_rate = 700
cat_amount, FF_amount = 100, 300

with open('./dataset/molar_mass.pickle', 'rb') as f:
    molar_mass = pickle.load(f)
    
precursor_map = {
    'Ca': 'Ca(NO3)2', 'Co': 'Co(NO3)2', 'Cu': 'Cu(NO3)2', 'Fe': 'Fe(NO3)3', 
    'In': 'In(SO3CF3)3',  'Ir': 'IrCl3', 'Ni': 'Ni(NO3)2',  'Pd': 'PdCl2', 
    'Pt': 'H2PtCl6', 'Re': 'NH4ReO4', 'Rh': 'RhCl3', 'Ru': 'RuCl3', 'Sn': 'SnCl4','Zn': 'Zn(NO3)2'} 

pd_rows_list = []
for t in temp_range:
    for p in pres_range:
        
        new_row = {col: 0 for col in dataset.columns[:-1]}
        
        AM_1 = 'Pd'
        precursor_1 = precursor_map[AM_1]
        new_row[AM_1] = 5.0
        new_row[precursor_1] = 1
        new_row[selected_support] = 95.0
        new_row[selected_preparation] = 1 
        new_row[selected_solvent] = solvent_amount
        new_row['Stirring rate (rpm)'] = stir_rate
        new_row['Catalyst amount (mg)'] = cat_amount
        new_row['Furfural (mg)'] = FF_amount
        new_row['Operating_temp'] = t
        new_row['Operating_pressure'] = p
        new_row['Operating_time'] = oper_time
        new_row['Calcination_temp'] = calc_temp
        new_row['Reduction_temp'] = reduc_temp
        new_row['Calcination_time'] = calc_time
        new_row['Reduction_time'] = reduc_time
        furfural_mmol   = float(FF_amount / molar_mass['Furfural'])
        AM1_percent =  5.0 * 0.01    
        subs_to_metal = float(furfural_mmol / (cat_amount * AM1_percent / molar_mass[AM_1]))
        subs_concentration = float(FF_amount / solvent_amount)
        new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
        new_row['Substrate concentration (mg/ml)'] = subs_concentration
        pd_rows_list.append(new_row)

Pd_baseline_df = pd.DataFrame(pd_rows_list)
X_features = dataset.columns.tolist()[:-1] 
y_pred = model.predict(Pd_baseline_df[X_features])
y_pred = np.clip(y_pred, 0, 100)
Pd_baseline_df['THFA_yield (%)'] = y_pred
Pd_baseline_df['Combination'] = 'Pd (5wt%)'

ni_rows_list = []
for t in temp_range:
    for p in pres_range:
        
        new_row = {col: 0 for col in dataset.columns[:-1]}
        
        AM_1, AM_2 = 'Ni', 'Re'
        precursor_1, precursor_2 = precursor_map[AM_1], precursor_map[AM_2]
        new_row[AM_1], new_row[AM_2] = 4.0, 1.0
        
        new_row[precursor_1], new_row[precursor_2] = 1, 1
        new_row[selected_support] = 95.0
        new_row[selected_preparation] = 1 
        new_row[selected_solvent] = solvent_amount
        
        new_row['Stirring rate (rpm)'] = stir_rate
        new_row['Catalyst amount (mg)'] = cat_amount
        new_row['Furfural (mg)'] = FF_amount
        
        new_row['Operating_temp'] = t
        new_row['Operating_pressure'] = p
        new_row['Operating_time'] = oper_time
        
        new_row['Calcination_temp'] = calc_temp
        new_row['Reduction_temp'] = reduc_temp
        
        new_row['Calcination_time'] = calc_time
        new_row['Reduction_time'] = reduc_time

        furfural_mmol   = float(FF_amount / molar_mass['Furfural'])
    
        AM1_percent, AM2_percent = 4.0 * 0.01, 1.0 * 0.01 
        
        subs_to_metal = float(furfural_mmol / (cat_amount * AM1_percent / molar_mass[AM_1] + cat_amount * AM2_percent / molar_mass[AM_2]))
        subs_concentration = float(FF_amount / solvent_amount)
            
        new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
        new_row['Substrate concentration (mg/ml)'] = subs_concentration
        
        ni_rows_list.append(new_row)

Ni_X_baseline_df = pd.DataFrame(ni_rows_list)
X_features = dataset.columns.tolist()[:-1] 
y_pred = model.predict(Ni_X_baseline_df[X_features])
y_pred = np.clip(y_pred, 0, 100)
Ni_X_baseline_df['THFA_yield (%)'] = y_pred
Ni_X_baseline_df['Combination'] = f'Ni (4wt%) / {AM_2} (1wt%)'

Ni_X_baseline_df['delta_y'] = (Ni_X_baseline_df['THFA_yield (%)'].values- Pd_baseline_df['THFA_yield (%)'].values)

heatmap_data = Ni_X_baseline_df.pivot_table(index='Operating_pressure', columns='Operating_temp', values='delta_y')

T = heatmap_data.columns.values.astype(float)
P = heatmap_data.index.values.astype(float)
TT, PP = np.meshgrid(T, P)
ZZ = heatmap_data.values

def build_and_predict_case(active_metals, combination_label):
    """
    active_metals example:
        {'Pd': 5.0}
        {'Ni': 4.0, 'Re': 1.0}
        {'Ni': 5.0}
    """
    rows = []

    for t in temp_range:
        for p in pres_range:
            new_row = {col: 0 for col in dataset.columns[:-1]}

            # active metal loading and precursor one-hot
            for metal, loading in active_metals.items():
                new_row[metal] = loading
                precursor = precursor_map[metal]
                new_row[precursor] = 1

            total_metal_loading = sum(active_metals.values())
            new_row[selected_support] = 100.0 - total_metal_loading
            new_row[selected_preparation] = 1
            new_row[selected_solvent] = solvent_amount

            new_row['Stirring rate (rpm)'] = stir_rate
            new_row['Catalyst amount (mg)'] = cat_amount
            new_row['Furfural (mg)'] = FF_amount

            new_row['Operating_temp'] = t
            new_row['Operating_pressure'] = p
            new_row['Operating_time'] = oper_time

            new_row['Calcination_temp'] = calc_temp
            new_row['Reduction_temp'] = reduc_temp
            new_row['Calcination_time'] = calc_time
            new_row['Reduction_time'] = reduc_time

            # substrate-to-metal ratio and substrate concentration
            furfural_mmol = float(FF_amount / molar_mass['Furfural'])

            metal_mmol = 0.0
            for metal, loading in active_metals.items():
                metal_fraction = loading * 0.01
                metal_mmol += cat_amount * metal_fraction / molar_mass[metal]

            subs_to_metal = float(furfural_mmol / metal_mmol)
            subs_concentration = float(FF_amount / solvent_amount)

            new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
            new_row['Substrate concentration (mg/ml)'] = subs_concentration

            rows.append(new_row)

    case_df = pd.DataFrame(rows)

    # Make sure columns are aligned with training features
    case_X = case_df.reindex(columns=X_features, fill_value=0)
    y_pred = model.predict(case_X)
    y_pred = np.clip(y_pred, 0, 100)

    case_df['THFA_yield (%)'] = y_pred
    case_df['Combination'] = combination_label

    return case_df

Pd_baseline_df = build_and_predict_case(
    active_metals={'Pd': 5.0},
    combination_label='5Pd'
)

pd_ref = Pd_baseline_df[
    ['Operating_temp', 'Operating_pressure', 'THFA_yield (%)']
].rename(columns={'THFA_yield (%)': 'THFA_yield_Pd (%)'})

non_noble_target = ['Co', 'Fe', 'Cu', 'Ca', 'Zn', 'Re']
gap_threshold = 0.0
heatmap_dict = {}
mean_yield_diff = {}
positive_region_fraction = {}
max_yield_diff = {}

for X in non_noble_target:
    label = f'1{X}-4Ni'

    NiX_df = build_and_predict_case(
        active_metals={'Ni': 4.0, X: 1.0},
        combination_label=label
    )

    NiX_df = NiX_df.merge(
        pd_ref,
        on=['Operating_temp', 'Operating_pressure'],
        how='left'
    )

    NiX_df['delta_y'] = (
        NiX_df['THFA_yield (%)'] - NiX_df['THFA_yield_Pd (%)']
    )

    heatmap_data = NiX_df.pivot_table(
        index='Operating_pressure',
        columns='Operating_temp',
        values='delta_y'
    )

    Z = heatmap_data.values

    heatmap_dict[label] = heatmap_data
    mean_yield_diff[label] = float(np.nanmean(Z))
    max_yield_diff[label] = float(np.nanmax(Z))
    positive_region_fraction[label] = float(np.nanmean(Z >= gap_threshold) * 100.0)

screening_metrics_df = pd.DataFrame({
    'Catalyst': list(mean_yield_diff.keys()),
    'Mean yield gap (%)': list(mean_yield_diff.values()),
    'Maximum yield gap (%)': list(max_yield_diff.values()),
    f'Positive-region fraction, ΔY >= {gap_threshold:g} (%)': list(positive_region_fraction.values())
})

heatmap_data = heatmap_dict['1Re-4Ni']

T = heatmap_data.columns.values.astype(float)
P = heatmap_data.index.values.astype(float)
TT, PP = np.meshgrid(T, P)
ZZ = heatmap_data.values

####################################################################################################################################
#figure
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 170 * MM_TO_INCH

FS_PANEL = 8.0
FS_LABEL = 6.5
FS_TICK = 5.5
FS_TEXT = 5.5
FS_VALUE = 5.2
FS_LEGEND = 6.0
FS_CBAR = 6.0
FS_CBAR_TICK = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 0.9,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

features = ['Operating_temp', 'Operating_pressure', 'Operating_time']
x_labels = ['Operating temperature (°C)', 'Operating pressure (bar)', 'Operating time (h)']
x_lims = [(50, 200), (0, 40), (0, 10)]
x_ticks = [[50, 100, 150, 200], [0, 10, 20, 30, 40], [0, 2, 4, 6, 8, 10]]
y_lim_pdp = (-12, 62)
y_ticks_pdp = [0, 20, 40, 60]

re_key = '1Re-4Ni'
heatmap_data = heatmap_dict[re_key]

T = heatmap_data.columns.to_numpy(dtype=float)
P = heatmap_data.index.to_numpy(dtype=float)
TT, PP = np.meshgrid(T, P)
ZZ = heatmap_data.to_numpy(dtype=float)

labels = [f'1{metal}-4Ni' for metal in non_noble_target if f'1{metal}-4Ni' in positive_region_fraction]

if len(labels) == 0:
    labels = list(positive_region_fraction.keys())
positive_values = np.array([positive_region_fraction[label] for label in labels], dtype=float)
mean_gap_values = np.array([mean_yield_diff[label] for label in labels], dtype=float)
display_labels = [label.replace('-', '–') for label in labels]

#set colors
COLOR_PD = '#CC4C4C'
COLOR_NI = '#4775B8'
COLOR_BAR = '#8FB6D6'
COLOR_RE = '#D98E04'
COLOR_RE_TEXT = '#B2182B'
COLOR_EDGE = '0.25'
COLOR_ZERO = '0.25'
heat_vmin = -20
heat_vmax = 15
custom_cmap = mcolors.LinearSegmentedColormap.from_list('potential_map',['#1F4E79', '#8FB6D6',  '#F7F7F7', '#E6C875', '#D76445', '#9C1515'])
heat_norm = mcolors.TwoSlopeNorm(vmin=heat_vmin, vcenter=0, vmax=heat_vmax)

def add_panel_label(ax, label):
    ax.text(
        0.00,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        ha='left',
        va='bottom',
        color='black',
        clip_on=False
    )


def style_axis(
    ax,
    grid_axis=None
):

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    ax.grid(False)

    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle=':',
            linewidth=0.35,
            color='0.82',
            alpha=0.8
        )

    ax.set_axisbelow(True)


def get_pd_grid(result):
    if 'grid_values' in result:
        return result['grid_values'][0]

    return result['values'][0]


def remove_contour_seams(contour_set):
    for collection in getattr(contour_set, 'collections', []):
        collection.set_edgecolor('face')
        collection.set_linewidth(0.0)
        collection.set_antialiased(False)
        collection.set_rasterized(True)


#figure layout
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

outer_gs = fig.add_gridspec(
    nrows=2, ncols=1, height_ratios=[0.82, 1.58],
    left=0.075, right=0.985, bottom=0.075, top=0.915, hspace=0.25)

top_gs = outer_gs[0].subgridspec(nrows=1, ncols=3, wspace=0.25)
bottom_gs = outer_gs[1].subgridspec(nrows=1, ncols=2, width_ratios=[2.15, 1.0], wspace=0.35)
heat_gs = bottom_gs[0].subgridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.038], wspace=0.07)
right_gs = bottom_gs[1].subgridspec(nrows=2, ncols=1, hspace=0.48)

ax1 = fig.add_subplot(top_gs[0, 0])
ax2 = fig.add_subplot(top_gs[0, 1])
ax3 = fig.add_subplot(top_gs[0, 2])
ax4 = fig.add_subplot(heat_gs[0, 0])
cax = fig.add_subplot(heat_gs[0, 1])
ax5 = fig.add_subplot(right_gs[0, 0])
ax6 = fig.add_subplot(right_gs[1, 0])
pdp_axes = [ax1, ax2, ax3]

#monometalic Pd vs Ni-based non-noble-metal catalysts
for idx, (ax, feature, x_label, x_lim,ticks) in enumerate(zip(pdp_axes, features, x_labels, x_lims, x_ticks)):

    pd_result = partial_dependence(
        estimator=model,
        X=Pd_only_df,
        features=[feature],
        kind='average',
        method='brute',
        grid_resolution=100,
        percentiles=(0.0, 1.0)
    )

    ni_result = partial_dependence(
        estimator=model,
        X=Ni_only_df,
        features=[feature],
        kind='average',
        method='brute',
        grid_resolution=100,
        percentiles=(0.0, 1.0)
    )

    pd_grid = get_pd_grid(pd_result)
    ni_grid = get_pd_grid(ni_result)

    pd_average = np.asarray(pd_result['average'][0], dtype=float)
    ni_average = np.asarray(ni_result['average'][0], dtype=float)

    ax.plot(pd_grid, pd_average, color=COLOR_PD, linewidth=1.2, zorder=3)
    ax.plot(ni_grid, ni_average, color=COLOR_NI, linewidth=1.2, zorder=3)

    #rug plot (dataset distribution)
    sns.rugplot(x=Pd_only_df[feature], ax=ax, color=COLOR_PD,
        alpha=0.55, height=0.035, linewidth=0.45)
    sns.rugplot(x=Ni_only_df[feature], ax=ax, color=COLOR_NI,
        alpha=0.55, height=0.035, linewidth=0.45)
    ax.axhline(y=0, color='0.45', linewidth=0.55, linestyle='-', zorder=1)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim_pdp)
    ax.set_xticks(ticks)
    ax.set_yticks(y_ticks_pdp)
    ax.set_xlabel(x_label, fontsize=FS_LABEL, labelpad=3)

    if idx == 0:
        ax.set_ylabel('Partial dependence value', fontsize=FS_LABEL, labelpad=3)
    else:
        ax.set_ylabel('')

    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_panel_label(ax, chr(ord('a') + idx))
    style_axis(ax, grid_axis=None)

legend_handles = [Line2D([0], [0], color=COLOR_PD, linewidth=1.2,
                         label='Monometallic Pd catalysts'),
                  Line2D([0], [0], color=COLOR_NI, linewidth=1.2,
                         label='Ni-based non-noble metal catalysts')]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.972),
    ncol=2,
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=2.3,
    handletextpad=0.5,
    columnspacing=1.7,
    borderaxespad=0
)

#Yield gap heatmap
levels = np.linspace(heat_vmin, heat_vmax, 141)
contour = ax4.contourf(TT, PP, ZZ, levels=levels, cmap=custom_cmap, norm=heat_norm, extend='both', antialiased=False)
remove_contour_seams(contour)

finite_zz = ZZ[np.isfinite(ZZ)]

if (finite_zz.size > 0 and np.nanmin(finite_zz) < 0 and np.nanmax(finite_zz) > 0):
    zero_contour = ax4.contour(TT, PP, ZZ,
        levels=[0], colors='black', linewidths=0.8, linestyles='--', zorder=4)

    ax4.clabel(zero_contour, inline=True, inline_spacing=3, fontsize=FS_TEXT,
        fmt={0: r'$\Delta Y = 0$'})

ax4.set_xlim(np.nanmin(T), np.nanmax(T))
ax4.set_ylim(np.nanmin(P), np.nanmax(P))
ax4.set_xticks([120, 140, 160, 180, 200])
ax4.set_yticks([10, 15, 20, 25, 30, 35, 40])
ax4.set_xlabel('Operating temperature (°C)', fontsize=FS_LABEL, labelpad=3)
ax4.set_ylabel('Operating pressure (bar)',
    fontsize=FS_LABEL,
    labelpad=3
)

ax4.tick_params(
    axis='both',
    labelsize=FS_TICK
)

add_panel_label(
    ax4,
    'd'
)

style_axis(
    ax4,
    grid_axis=None
)


# =============================================================================
# 12. Heatmap colorbar
# =============================================================================
cbar = fig.colorbar(
    contour,
    cax=cax,
    extend='both'
)

cbar.set_ticks(
    [-20, -15, -10, -5, 0, 5, 10, 15]
)

cbar.set_label(
    r'Yield difference, $\Delta Y$ (%)',
    fontsize=FS_CBAR,
    labelpad=4
)

cbar.ax.tick_params(
    axis='y',
    labelsize=FS_CBAR_TICK,
    direction='out',
    width=0.6,
    length=2.5,
    pad=2
)

cbar.outline.set_linewidth(0.6)

#Positive region fraction
x = np.arange(len(labels))

bar_colors_positive = [COLOR_RE
    if label == re_key
    else COLOR_BAR
    for label in labels]

bars_positive = ax5.bar(
    x=x, height=positive_values, width=0.70,
    color=bar_colors_positive,
    edgecolor=COLOR_EDGE,
    linewidth=0.35,
    zorder=2
)

for bar, label, value in zip(
    bars_positive,
    labels,
    positive_values
):
    ax5.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 2.0,

        f'{value:.1f}',

        ha='center',
        va='bottom',

        fontsize=FS_VALUE,
        fontweight=(
            'bold'
            if label == re_key
            else 'normal'
        ),

        color=(
            COLOR_RE_TEXT
            if label == re_key
            else 'black'
        ),

        clip_on=False
    )


ax5.set_ylim(
    0,
    105
)

ax5.set_xticks(x)

ax5.set_xticklabels(
    display_labels,
    rotation=35,
    ha='center',
    rotation_mode='anchor',
    fontsize=FS_TICK
)

ax5.set_ylabel(
    r'Region with $\Delta Y \geq 0$ (%)',
    fontsize=FS_LABEL,
    labelpad=3
)

ax5.set_yticks(
    [0, 20, 40, 60, 80, 100]
)

for tick, label in zip(
    ax5.get_xticklabels(),
    labels
):
    if label == re_key:
        tick.set_color(
            COLOR_RE_TEXT
        )
        tick.set_fontweight(
            'bold'
        )

add_panel_label(
    ax5,
    'e'
)

style_axis(
    ax5,
    grid_axis=None
)

ax5.tick_params(
    axis='x',
    pad=6
)

#Mean yield gap
bar_colors_mean = [
    COLOR_RE
    if label == re_key
    else COLOR_BAR
    for label in labels
]

bars_mean = ax6.bar(
    x=x,
    height=mean_gap_values,
    width=0.70,

    color=bar_colors_mean,
    edgecolor=COLOR_EDGE,
    linewidth=0.35,
    zorder=2
)

ax6.axhline(
    y=0,
    color=COLOR_ZERO,
    linestyle='--',
    linewidth=0.75,
    zorder=3
)


for bar, label, value in zip(
    bars_mean,
    labels,
    mean_gap_values
):
    text_offset = (
        1.5
        if value >= 0
        else -1.5
    )

    ax6.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + text_offset,

        f'{value:.1f}',

        ha='center',

        va=(
            'bottom'
            if value >= 0
            else 'top'
        ),

        fontsize=FS_VALUE,
        fontweight=(
            'bold'
            if label == re_key
            else 'normal'
        ),

        color=(
            COLOR_RE_TEXT
            if label == re_key
            else 'black'
        ),

        clip_on=False
    )


mean_min = float(
    np.nanmin(mean_gap_values)
)

mean_max = float(
    np.nanmax(mean_gap_values)
)

lower_limit = min(
    -5,
    np.floor(
        (mean_min - 5) / 10
    ) * 10
)

upper_limit = max(
    10,
    np.ceil(
        (mean_max + 3) / 5
    ) * 5
)

ax6.set_ylim(lower_limit, upper_limit)
ax6.set_xticks(x)
ax6.set_xticklabels(display_labels, rotation=35, ha='center', rotation_mode='anchor', fontsize=FS_TICK)
ax6.set_ylabel(r'Mean $\Delta Y$ (%)', fontsize=FS_LABEL, labelpad=3)
ax6.yaxis.set_major_locator(MaxNLocator(nbins=5))

for tick, label in zip(ax6.get_xticklabels(), labels):
    if label == re_key:
        tick.set_color(COLOR_RE_TEXT)
        tick.set_fontweight('bold')

add_panel_label(ax6, 'f')
style_axis(ax6, grid_axis=None)
ax6.tick_params(axis='x', pad=6)

# plt.savefig('./figure_3.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_3.pdf', dpi=600, bbox_inches='tight')
plt.show()
#%% Fig. 4-(a)
data = {
    'Catalyst': [
        '1Ru–4Ni', '1Pd–4Ni', '1Ir–4Ni', '1Pt–4Ni', '1Rh–4Ni', '5Pd',
        '1Co–4Ni', '1Fe–4Ni', '1Cu–4Ni', '1Ca–4Ni',
        '1Zn–4Ni', '1Re–4Ni', '5Ni', '5Re'
    ],
    'Experimental': [
        80.45, 73.34, 73.89, 55.48, 66.48, 87.89,
        10.83, 49.53, 37.57, 31.37, 26.25, 84.26,
        32.58, 15.26
    ],
    'Predicted': [
        67.25, 73.04, 73.50, 55.19, 53.42, 78.86,
        11.08, 49.44, 37.50, 41.21, 39.57, 80.12,
        32.97, 49.25
    ],
    'noble': [
        True, True, True, True, True, True,
        False, False, False, False, False, False,
        False, False
    ]
}

df = pd.DataFrame(data)

noble_idx = df.index[df['noble']].to_numpy()
noble_free_idx = df.index[~df['noble']].to_numpy()
#####################################################################################################################
MM_TO_INCH = 1 / 25.4

# Full-width figure
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 115 * MM_TO_INCH

# Nature figure typography
# Panel label: 8 pt
# All other text: 5–7 pt
FS_PANEL = 12.0
FS_AXIS = 10.0
FS_GROUP = 9.5
FS_TICK = 7
FS_VALUE = 6.0
FS_LEGEND = 7.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_AXIS,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'legend.fontsize': FS_LEGEND,
    'legend.frameon': False,

    # Editable embedded text in vector files
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})


# =============================================================================
# 2. Accessible colours
# =============================================================================
# Colour-blind-accessible palette
COLOR_EXP_NOBLE = '#0072B2'     # dark blue
COLOR_PRED_NOBLE = '#56B4E9'    # sky blue
COLOR_EXP_FREE = '#D55E00'      # vermillion
COLOR_PRED_FREE = '#E69F00'     # orange


# =============================================================================
# 3. Figure
# =============================================================================
fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

x = np.arange(len(df))
bar_width = 0.38


# =============================================================================
# 4. Bars
# =============================================================================
bars_exp_noble = ax.bar(
    x[noble_idx] - bar_width / 2,
    df.loc[noble_idx, 'Experimental'],
    width=bar_width,
    color=COLOR_EXP_NOBLE,
    edgecolor='black',
    linewidth=0.35,
    label='Experimental (noble-metal)',
    zorder=3
)

bars_pred_noble = ax.bar(
    x[noble_idx] + bar_width / 2,
    df.loc[noble_idx, 'Predicted'],
    width=bar_width,
    color=COLOR_PRED_NOBLE,
    edgecolor='black',
    linewidth=0.35,
    label='Predicted (noble-metal)',
    zorder=3
)

bars_exp_free = ax.bar(
    x[noble_free_idx] - bar_width / 2,
    df.loc[noble_free_idx, 'Experimental'],
    width=bar_width,
    color=COLOR_EXP_FREE,
    edgecolor='black',
    linewidth=0.35,
    label='Experimental (noble-metal-free)',
    zorder=3
)

bars_pred_free = ax.bar(
    x[noble_free_idx] + bar_width / 2,
    df.loc[noble_free_idx, 'Predicted'],
    width=bar_width,
    color=COLOR_PRED_FREE,
    edgecolor='black',
    linewidth=0.35,
    label='Predicted (noble-metal-free)',
    zorder=3
)


# =============================================================================
# 5. Bar-value labels
# =============================================================================
SHOW_VALUES = True


def add_bar_labels(axis, bars):
    if not SHOW_VALUES:
        return

    for bar in bars:
        height = bar.get_height()

        if not np.isfinite(height):
            continue

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=FS_VALUE,
            fontweight='normal',
            color='black',
            rotation=0,
            clip_on=False
        )


add_bar_labels(ax, bars_exp_noble)
add_bar_labels(ax, bars_pred_noble)
add_bar_labels(ax, bars_exp_free)
add_bar_labels(ax, bars_pred_free)


# =============================================================================
# 6. Catalyst-group separation
# =============================================================================
group_boundary = len(noble_idx) - 0.5

# Background shading is omitted because it is not essential.
# Use only a simple group-separation line.
ax.axvline(
    group_boundary,
    color='0.55',
    linewidth=0.6,
    linestyle='-',
    zorder=2
)

noble_center = (
    noble_idx.min() + noble_idx.max()
) / 2

free_center = (
    noble_free_idx.min() + noble_free_idx.max()
) / 2


# Group labels: Nature recommends avoiding coloured text
ax.text(
    noble_center,
    1.015,
    'Noble-metal catalysts',
    transform=ax.get_xaxis_transform(),
    ha='center',
    va='bottom',
    fontsize=FS_GROUP,
    fontweight='bold',
    color='black',
    clip_on=False
)

ax.text(
    free_center,
    1.015,
    'Noble-metal-free catalysts',
    transform=ax.get_xaxis_transform(),
    ha='center',
    va='bottom',
    fontsize=FS_GROUP,
    fontweight='bold',
    color='black',
    clip_on=False
)


# =============================================================================
# 7. Axes
# =============================================================================
ax.set_xticks(x)

ax.set_xticklabels(
    df['Catalyst'],
    rotation=35,
    ha='right',
    rotation_mode='anchor',
    fontsize=FS_TICK
)

ax.set_xlabel(
    'Catalyst',
    fontsize=FS_AXIS,
    labelpad=5
)

ax.set_ylabel(
    'THFA yield (%)',
    fontsize=FS_AXIS,
    labelpad=4
)

ax.set_xlim(
    -0.65,
    len(df) - 0.35
)

# Dynamic upper limit with enough room for value labels
all_values = np.concatenate([
    df.loc[noble_idx, 'Experimental'].to_numpy(dtype=float),
    df.loc[noble_idx, 'Predicted'].to_numpy(dtype=float),
    df.loc[noble_free_idx, 'Experimental'].to_numpy(dtype=float),
    df.loc[noble_free_idx, 'Predicted'].to_numpy(dtype=float)
])

max_bar_value = np.nanmax(all_values)

y_upper = max(
    105,
    np.ceil((max_bar_value + 5) / 5) * 5
)

ax.set_ylim(
    0,
    y_upper
)

ax.set_yticks(
    np.arange(
        0,
        min(y_upper, 100) + 1,
        20
    )
)

ax.tick_params(
    axis='x',
    labelsize=FS_TICK,
    direction='out',
    width=0.6,
    length=2.5,
    pad=3
)

ax.tick_params(
    axis='y',
    labelsize=FS_TICK,
    direction='out',
    width=0.6,
    length=2.5,
    pad=2
)

# Nature guide recommends avoiding background gridlines
ax.grid(False)
ax.set_axisbelow(True)


# =============================================================================
# 8. Full rectangular border
# =============================================================================
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.6)

# =============================================================================
# 10. Legend
# =============================================================================
legend_handles = [
    Patch(
        facecolor=COLOR_EXP_NOBLE,
        edgecolor='black',
        linewidth=0.35,
        label='Experimental (noble-metal)'
    ),
    Patch(
        facecolor=COLOR_PRED_NOBLE,
        edgecolor='black',
        linewidth=0.35,
        label='Predicted (noble-metal)'
    ),
    Patch(
        facecolor=COLOR_EXP_FREE,
        edgecolor='black',
        linewidth=0.35,
        label='Experimental (noble-metal-free)'
    ),
    Patch(
        facecolor=COLOR_PRED_FREE,
        edgecolor='black',
        linewidth=0.35,
        label='Predicted (noble-metal-free)'
    )
]

ax.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=4,                    # 한 줄로 배치
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=1.15,
    handleheight=0.8,
    handletextpad=0.35,
    columnspacing=0.90,
    labelspacing=0.0,
    borderaxespad=0
)

fig.subplots_adjust(
    left=0.090,
    right=0.990,
    bottom=0.245,
    top=0.780
)

# plt.savefig('./figure_4a.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_4a.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Fig. 6 - MSP comparison

# ── Data ──────────────────────────────────────────────────────────────────
catalysts = ["Ni–Re", "Pd",     "Ni–Ru", "Ni–Pd", "Ni–Ir",  "Ni–Pt", "Ni–Rh"]
MSP       = np.array([3181.42, 4021.61 , 3439.52, 3873.25, 4523.35, 5175.07, 5366.40])
yield_exp = np.array([84.26,   86.89,   80.45,   73.34,   73.89,   55.48,   66.48])
RMV       = np.array([26.60,  1768.29,  222.44,  354.26, 1415.23,  386.41, 1865.34])
MARKET    = 3500

# ── Normalize x only by mean ──────────────────────────────────────────────
avg_yield = yield_exp.mean()   # ~74.40  → y 기준선으로만 사용
avg_rmv   = RMV.mean()         # ~862.65

rmv_norm  = RMV / avg_rmv      # x축: normalized

# ── Colors / markers ──────────────────────────────────────────────────────
C_PD = "#EE854A"
C_NIBIMETAL = "#74ADD1"
C_NIRE = "#2166AC"
C_MARKET = "#C0392B"

def get_color(catalyst):
    if catalyst == "Ni–Re":
        return C_NIRE
    if catalyst == "Pd":
        return C_PD
    return C_NIBIMETAL

def get_marker(catalyst):
    if catalyst == "Ni–Re":
        return "*"
    if catalyst == "Pd":
        return "o"
    return "s"

colors = [get_color(c) for c in catalysts]
#################################################################################################################
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 100 * MM_TO_INCH

FS_PANEL = 8.0
FS_REGION = 6.5
FS_LABEL = 6.5
FS_TICK = 5.5
FS_TEXT = 5.8
FS_LEGEND = 5.8

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [
        'Arial',
        'Helvetica',
        'Liberation Sans',
        'DejaVu Sans'
    ],

    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})


# =============================================================================
# 2. Data
# 실제 분석값으로 교체
# =============================================================================
plot_df = pd.DataFrame({
    'Catalyst': [
        'Ni–Re',
        'Ni–Ru',
        'Ni–Pd',
        'Ni–Pt',
        'Ni–Ir',
        'Ni–Rh',
        'Pd'
    ],

    'Experimental_yield': [
        84.3,
        80.5,
        73.4,
        55.5,
        73.8,
        66.5,
        86.8
    ],

    'Relative_cost': [
        0.031,
        0.255,
        0.410,
        0.450,
        1.65,
        2.15,
        2.02
    ],

       'MSP': [
        3181.4,
        3439.5,
        3873.3,
        5175.1,
        4523.4,
        5366.4,
        4021.61
    ],

    'Type': [
        'Candidate',
        'Ni-based',
        'Ni-based',
        'Ni-based',
        'Ni-based',
        'Ni-based',
        'Pd'
    ]
})


# =============================================================================
# 3. Thresholds
# =============================================================================
# Panel b
mean_yield = 74.3
mean_relative_cost = 1.0

# Panel c
high_yield_threshold = 80.0
market_price_low = 3000
market_price_high = 5000

# =============================================================================
# 4. Colors
# =============================================================================
COLOR_CANDIDATE = '#2C6DB2'
COLOR_NI = '#6BAED6'
COLOR_PD = '#F28E5B'

COLOR_TARGET_GREEN = '#5B8C5A'
COLOR_HIGH_COST = '#9A6A36'
COLOR_LOW_COST = '#536D8D'
COLOR_POOR = '#8A5A5A'

COLOR_TARGET_BLUE = '#2C6DB2'
COLOR_MARKET = '#C43C35'

COLOR_REFERENCE = '0.45'
COLOR_EDGE = '0.25'


# =============================================================================
# 5. Figure layout
# =============================================================================
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

ax_b, ax_c = axes

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.225,
    top=0.930,
    wspace=0.28
)


# =============================================================================
# 6. Common plotting function
# =============================================================================
def draw_catalyst_markers(ax, x_column):
    """
    Ni–Re candidate, Ni-based catalysts, and Pd benchmark를
    서로 다른 marker로 표시합니다.
    """

    ni_data = plot_df[
        plot_df['Type'] == 'Ni-based'
    ]

    candidate_data = plot_df[
        plot_df['Type'] == 'Candidate'
    ]

    pd_data = plot_df[
        plot_df['Type'] == 'Pd'
    ]

    # Ni-based bimetallic catalysts
    ax.scatter(
        ni_data[x_column],
        ni_data['Experimental_yield'],
        marker='s',
        s=42,
        facecolor=COLOR_NI,
        edgecolor=COLOR_EDGE,
        linewidth=0.55,
        zorder=4
    )

    # Ni–Re candidate
    ax.scatter(
        candidate_data[x_column],
        candidate_data['Experimental_yield'],
        marker='*',
        s=125,
        facecolor=COLOR_CANDIDATE,
        edgecolor=COLOR_EDGE,
        linewidth=0.65,
        zorder=5
    )

    # Pd benchmark
    ax.scatter(
        pd_data[x_column],
        pd_data['Experimental_yield'],
        marker='o',
        s=52,
        facecolor=COLOR_PD,
        edgecolor=COLOR_EDGE,
        linewidth=0.60,
        zorder=5
    )


def style_axis(ax):
    """
    공통 축 스타일.
    """

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    ax.set_axisbelow(True)


# =============================================================================
# 7. Panel b: Experimental yield vs relative catalyst cost
# =============================================================================
b_xmin = 0.015
b_xmax = 3.5
b_ymin = 48
b_ymax = 95


# -----------------------------------------------------------------------------
# 7.1 Decision-region shading
# -----------------------------------------------------------------------------
# High yield / low cost: 핵심 목표 영역
ax_b.add_patch(
    Rectangle(
        (b_xmin, mean_yield),
        mean_relative_cost - b_xmin,
        b_ymax - mean_yield,
        facecolor=COLOR_TARGET_GREEN,
        edgecolor='none',
        alpha=0.085,
        zorder=0
    )
)

# High yield / high cost
ax_b.add_patch(
    Rectangle(
        (mean_relative_cost, mean_yield),
        b_xmax - mean_relative_cost,
        b_ymax - mean_yield,
        facecolor=COLOR_HIGH_COST,
        edgecolor='none',
        alpha=0.030,
        zorder=0
    )
)

# Low yield / low cost
ax_b.add_patch(
    Rectangle(
        (b_xmin, b_ymin),
        mean_relative_cost - b_xmin,
        mean_yield - b_ymin,
        facecolor=COLOR_LOW_COST,
        edgecolor='none',
        alpha=0.020,
        zorder=0
    )
)

# Low yield / high cost
ax_b.add_patch(
    Rectangle(
        (mean_relative_cost, b_ymin),
        b_xmax - mean_relative_cost,
        mean_yield - b_ymin,
        facecolor=COLOR_POOR,
        edgecolor='none',
        alpha=0.025,
        zorder=0
    )
)


# -----------------------------------------------------------------------------
# 7.2 Mean reference lines
# -----------------------------------------------------------------------------
ax_b.axhline(
    mean_yield,
    color=COLOR_REFERENCE,
    linestyle='--',
    linewidth=0.8,
    zorder=2
)

ax_b.axvline(
    mean_relative_cost,
    color=COLOR_REFERENCE,
    linestyle='--',
    linewidth=0.8,
    zorder=2
)


# -----------------------------------------------------------------------------
# 7.3 Catalyst markers
# -----------------------------------------------------------------------------
draw_catalyst_markers(
    ax=ax_b,
    x_column='Relative_cost'
)


# -----------------------------------------------------------------------------
# 7.4 Catalyst labels
# -----------------------------------------------------------------------------
label_offsets_b = {
    'Ni–Re': (0, 9),
    'Ni–Ru': (-1, 8),
    'Ni–Pd': (0, -9),
    'Ni–Pt': (0, -9),
    'Ni–Ir': (0, 8),
    'Ni–Rh': (0, 8),
    'Pd': (-0.5, 8)
}

for _, row in plot_df.iterrows():

    catalyst = row['Catalyst']
    dx, dy = label_offsets_b[catalyst]

    ax_b.annotate(
        catalyst,
        xy=(
            row['Relative_cost'],
            row['Experimental_yield']
        ),
        xytext=(dx, dy),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=FS_TEXT,
        color='black',
        zorder=6
    )


# -----------------------------------------------------------------------------
# 7.5 Region labels
# -----------------------------------------------------------------------------
ax_b.text(
    0.018,
    93.5,
    'High yield / low cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#496F49',
    ha='left',
    va='top'
)

ax_b.text(
    3.15,
    94.0,
    'High yield /\nhigh cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#7A582C',
    ha='right',
    va='top',
    linespacing=0.95
)

ax_b.text(
    0.018,
    49.5,
    'Low yield / low cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#4C6482',
    ha='left',
    va='bottom'
)

ax_b.text(
    3.15,
    49.5,
    'Low yield /\nhigh cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#805454',
    ha='right',
    va='bottom',
    linespacing=0.95
)


# -----------------------------------------------------------------------------
# 7.6 Reference labels
# -----------------------------------------------------------------------------
ax_b.text(
    0.017,
    mean_yield + 0.7,
    'Mean yield',
    fontsize=FS_TEXT,
    color=COLOR_REFERENCE,
    ha='left',
    va='bottom'
)

ax_b.text(
    mean_relative_cost * 1.035,
    51.0,
    'Mean cost',
    fontsize=FS_TEXT,
    color=COLOR_REFERENCE,
    rotation=90,
    ha='left',
    va='bottom'
)


# -----------------------------------------------------------------------------
# 7.7 Axes
# -----------------------------------------------------------------------------
ax_b.set_xscale('log')

ax_b.set_xlim(
    b_xmin,
    b_xmax
)

ax_b.set_ylim(
    b_ymin,
    b_ymax
)

ax_b.set_xticks(
    [0.03, 0.1, 0.3, 1, 3]
)

ax_b.set_xticklabels(
    ['0.03', '0.1', '0.3', '1', '3']
)

ax_b.set_yticks(
    [50, 60, 70, 80, 90]
)

ax_b.set_xlabel(
    'Relative catalyst raw-material cost',
    fontsize=FS_LABEL,
    labelpad=4
)

ax_b.set_ylabel(
    'Experimental THFA yield (%)',
    fontsize=FS_LABEL,
    labelpad=4
)

style_axis(ax_b)


# =============================================================================
# 8. Panel c: Experimental yield vs MSP
# =============================================================================
c_xmin = 2900
c_xmax = 5700
c_ymin = 48
c_ymax = 95


# -----------------------------------------------------------------------------
# 8.1 Target region
# -----------------------------------------------------------------------------
ax_c.axvspan(
    market_price_low,
    market_price_high,
    facecolor=COLOR_MARKET,
    edgecolor='none',
    alpha=0.055,
    zorder=0
)


# -----------------------------------------------------------------------------
# 8.2 Market-price reference
# -----------------------------------------------------------------------------
for x_bound in (market_price_low, market_price_high):
    ax_c.axvline(
        x_bound,
        color=COLOR_MARKET,
        linestyle='--',
        linewidth=0.8,
        alpha=0.85,
        zorder=2
    )
# -----------------------------------------------------------------------------
# 8.3 Catalyst markers
# -----------------------------------------------------------------------------
draw_catalyst_markers(
    ax=ax_c,
    x_column='MSP'
)


# -----------------------------------------------------------------------------
# 8.4 Catalyst labels and leader lines
# -----------------------------------------------------------------------------
label_offsets_c = {
    'Ni–Re': (0, 10),
    'Ni–Ru': (0, 8),
    'Ni–Pd': (0, -9),
    'Ni–Pt': (0, 8),
    'Ni–Ir': (0, 8),
    'Ni–Rh': (0, 8),
    'Pd': (0, 8)
}

leader_line_catalysts = set()

for _, row in plot_df.iterrows():

    catalyst = row['Catalyst']
    dx, dy = label_offsets_c[catalyst]

    arrowprops = None

    if catalyst in leader_line_catalysts:
        arrowprops = {
            'arrowstyle': '-',
            'color': '0.55',
            'linewidth': 0.55,
            'shrinkA': 1,
            'shrinkB': 3
        }

    ax_c.annotate(
        catalyst,
        xy=(
            row['MSP'],
            row['Experimental_yield']
        ),
        xytext=(dx, dy),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=FS_TEXT,
        color='black',
        arrowprops=arrowprops,
        zorder=6
    )


# -----------------------------------------------------------------------------
# 8.5 Target-region and market-price labels
# -----------------------------------------------------------------------------
ax_c.axhline(
    high_yield_threshold,
    color=COLOR_REFERENCE,
    linestyle='--',
    linewidth=0.8,
    zorder=2
)
ax_c.text(
    (market_price_low + market_price_high) / 2,
    93.5,
    'THFA market price range (3,000–5,000 $/ton)',
    fontsize=FS_REGION,
    fontweight='bold',
    color=COLOR_MARKET,
    ha='center',
    va='top'
)


# -----------------------------------------------------------------------------
# 8.6 Axes
# -----------------------------------------------------------------------------
ax_c.set_xlim(
    c_xmin,
    c_xmax
)

ax_c.set_ylim(
    c_ymin,
    c_ymax
)

ax_c.set_xticks(
    [3000, 3500, 4000, 4500, 5000, 5500]
)

ax_c.set_yticks(
    [50, 60, 70, 80, 90]
)

ax_c.set_xlabel(
    'Minimum selling price ($/ton THFA)',
    fontsize=FS_LABEL,
    labelpad=4
)

ax_c.set_ylabel(
    'Experimental THFA yield (%)',
    fontsize=FS_LABEL,
    labelpad=4
)

style_axis(ax_c)


# =============================================================================
# 9. Panel labels
# =============================================================================
ax_b.text(
    0.00,
    1.02,
    'b',
    transform=ax_b.transAxes,
    fontsize=FS_PANEL,
    fontweight='bold',
    ha='left',
    va='bottom',
    clip_on=False
)

ax_c.text(
    0.00,
    1.02,
    'c',
    transform=ax_c.transAxes,
    fontsize=FS_PANEL,
    fontweight='bold',
    ha='left',
    va='bottom',
    clip_on=False
)


# =============================================================================
# 10. Shared legend
# =============================================================================
legend_handles = [
    Line2D(
        [0],
        [0],
        marker='*',
        linestyle='none',
        markersize=8.5,
        markerfacecolor=COLOR_CANDIDATE,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.6,
        label='Ni–Re candidate'
    ),

    Line2D(
        [0],
        [0],
        marker='s',
        linestyle='none',
        markersize=5.5,
        markerfacecolor=COLOR_NI,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.6,
        label='Ni-based bimetallic catalysts'
    ),

    Line2D(
        [0],
        [0],
        marker='o',
        linestyle='none',
        markersize=5.8,
        markerfacecolor=COLOR_PD,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.6,
        label='Pd benchmark'
    )
]

fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.095),
    ncol=3,
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=1.0,
    handletextpad=0.45,
    columnspacing=1.6,
    borderaxespad=0
)

# plt.savefig('./figure_6_bc.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_6_bc.pdf', dpi=600, bbox_inches='tight') 
# fig.savefig('./figure_6_bc.svg')   
plt.show()

#%% Supplementary figures - Fig. 1 (Dataset distribution)

active_counts = (dataset[active_metal].gt(0).sum().sort_values(ascending=False))
active_counts = active_counts[active_counts > 0]

support_counts = (dataset[cat_support].gt(0).sum().sort_values(ascending=False))
support_counts = support_counts[support_counts > 0]

precursor_counts = ( dataset[precursor].gt(0).sum().sort_values(ascending=False))
precursor_counts = precursor_counts[precursor_counts > 0]

precursor_counts = precursor_counts.rename(index={'Unknown_precursor': 'Unknown precursor'})


preparation_counts = (dataset[preparation].gt(0).sum().sort_values(ascending=False))
preparation_counts = preparation_counts[preparation_counts > 0]

preparation_counts = preparation_counts.rename(index={'Unknown_preparation': 'Unknown preparation'})

solvent_counts = (dataset[solvent].gt(0).sum().sort_values(ascending=False))
solvent_counts = solvent_counts[solvent_counts > 0]

yield_col = 'THFA_yield (%)'
yield_data = dataset[yield_col].dropna().astype(float)



MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 150 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 6.0
FS_SMALL = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,

    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_SMALL,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 0.9,

    # PDF에서 텍스트 편집 가능
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

# =============================================================================
# 3. Plotting helpers
# =============================================================================
def set_panel_header(ax, panel_label, title):
    """
    패널 문자와 패널 제목을 동일한 높이에 배치합니다.
    """

    ax.text(
        0.00,
        1.045,
        panel_label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=FS_PANEL,
        fontweight='bold',
        fontstyle='normal',
        color='black',
        clip_on=False
    )

    # Subplot title
    ax.text(
        0.105,
        1.045,
        title,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=FS_TITLE,
        fontweight='normal',
        fontstyle='normal',
        color='black',
        clip_on=False
    )


def style_axis(ax, grid_axis=None):
    """
    모든 패널에 공통적인 NCE 스타일을 적용합니다.
    """

    # Full rectangular box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        width=0.6,
        length=2.5
    )

    # Background grid 제거
    ax.grid(False)

    ax.set_axisbelow(True)


def plot_bar(
    ax,
    series,
    panel_label,
    title,
    color,
    top_n=15
):
    """
    범주별 데이터 개수를 나타내는 수평 막대그래프입니다.
    """

    s = series.copy()

    if top_n is not None:
        s = s.head(top_n)

    # barh에서 가장 큰 값이 위쪽에 오도록 오름차순으로 재정렬
    s = s.sort_values(ascending=True)

    ax.barh(
        y=s.index,
        width=s.values,
        height=0.72,
        color=color,
        edgecolor='none',
        zorder=2
    )

    set_panel_header(
        ax=ax,
        panel_label=panel_label,
        title=title
    )

    ax.set_xlabel('Count', fontsize=FS_LABEL)
    ax.set_ylabel('')

    ax.tick_params(
        axis='x',
        labelsize=FS_TICK,
        pad=2
    )

    ax.tick_params(
        axis='y',
        labelsize=FS_SMALL,
        length=0,
        pad=2
    )

    # Count 축은 정수 눈금 사용
    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=5,
            integer=True,
            min_n_ticks=3
        )
    )

    ax.margins(y=0.025)

    style_axis(
        ax,
        grid_axis='x'
    )


# =============================================================================
# 4. Figure layout
# =============================================================================
fig = plt.figure(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=3,

    left=0.075,
    right=0.990,
    bottom=0.085,
    top=0.955,

    # 긴 범주명이 옆 패널과 겹치지 않도록 확보
    wspace=0.60,
    hspace=0.35
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]),
     fig.add_subplot(gs[0, 1]),
     fig.add_subplot(gs[0, 2])],

    [fig.add_subplot(gs[1, 0]),
     fig.add_subplot(gs[1, 1]),
     fig.add_subplot(gs[1, 2])]
])


# =============================================================================
# 5. Categorical statistics
# =============================================================================
plot_bar(
    axes[0, 0],
    active_counts,
    panel_label='a',
    title='',
    color='#4C78A8',
    top_n=15
)

plot_bar(
    axes[0, 1],
    precursor_counts,
    panel_label='b',
    title='',
    color='#54A24B',
    top_n=15
)

plot_bar(
    axes[0, 2],
    support_counts,
    panel_label='c',
    title='',
    color='#F58518',
    top_n=15
)

plot_bar(
    axes[1, 0],
    preparation_counts,
    panel_label='d',
    title='',
    color='#E45756',
    top_n=15
)

plot_bar(
    axes[1, 1],
    solvent_counts,
    panel_label='e',
    title='',
    color='#B279A2',
    top_n=15
)


# =============================================================================
# 6. THFA yield distribution
# =============================================================================
ax_yield = axes[1, 2]

sns.histplot(
    data=yield_data,
    bins=np.linspace(0, 100, 21),
    kde=True,
    stat='count',
    ax=ax_yield,

    color='#72B7B2',
    edgecolor='black',
    linewidth=0.35,
    alpha=0.75,

    line_kws={
        'linewidth': 1.0
    }
)

set_panel_header(
    ax=ax_yield,
    panel_label='f',
    title=''
)

ax_yield.set_xlim(0, 100)

ax_yield.set_xlabel(
    'THFA yield (%)',
    fontsize=FS_LABEL
)

ax_yield.set_ylabel(
    'Frequency',
    fontsize=FS_LABEL
)

ax_yield.tick_params(
    axis='both',
    labelsize=FS_TICK
)

ax_yield.xaxis.set_major_locator(
    MaxNLocator(
        nbins=6,
        integer=True
    )
)

ax_yield.yaxis.set_major_locator(
    MaxNLocator(
        nbins=5,
        integer=True
    )
)

style_axis(
    ax_yield,
    grid_axis='y'
)

# plt.savefig('./figure_supp_1.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_1.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 2 (All parity plots (Load model result))

model_name = ['XGBoost (XGB)', 'CatBoost (CB)', 'Random forest (RF)', 
              'LightGBM (LGBM)', 'Decision tree (DT)', 'Linear regression (LR)',
              'Lasso regression (Lasso)', 'Support vector regression (SVR)', 'Ridge regression (Ridge)']

base_path = './hyperparameter_tuning/output/'
output_files = ['xgb_model_seed_23.json', 'catboost_model_seed_23.cbm', 'RF_model_seed23.pkl', 
                'lightGBM_model_seed23.txt', 'DT_model_seed23.pkl', 'lr_model_seed_23.pkl', 
                'lasso_model_seed_23.pkl', 'svr_model_seed_23.pkl', 'ridge_model_seed_23.pkl']

# Prediction results
pred_results = []

for name, file in zip(model_name, output_files):
    model_path = base_path + file
    X_test  = pd.read_csv('./dataset/ML_dataset_final_x_test.csv')
    
    if 'cbm' in file:
        model = CatBoostRegressor()
        model.load_model(model_path)

    if 'xgb' in file:
        model = XGBRegressor()
        model.load_model(model_path)
    
    if 'lightGBM' in file:
        model = lgb.Booster(model_file=f'./hyperparameter_tuning/output/lightGBM_model_seed{SEED}.txt')
    
    if 'DT' in file or 'RF' in file:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    if 'lr' in file:
        with open(f'./hyperparameter_tuning/output/lr_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/lr_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)

    if 'lasso' in file:
        with open(f'./hyperparameter_tuning/output/lasso_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/lasso_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)
        
    if 'ridge' in file:
        with open(f'./hyperparameter_tuning/output/ridge_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/ridge_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)
    
    if 'svr' in file:
        with open(f'./hyperparameter_tuning/output/svr_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/svr_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)
        
    y_test_pred  = np.clip(model.predict(X_test),  0, 100)
    
    r2   = r2_score(np.array(y_test), y_test_pred)
    rmse = root_mean_squared_error(np.array(y_test), y_test_pred)

    pred_results.append({
        'model_name': name,
        'y_pred': y_test_pred,
        'r2': r2,
        'rmse': rmse
    })

    print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

#%% Supplementary figures - Fig. 2 (All parity plots (Figure))

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 170 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 5.5
FS_TEXT = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TEXT,
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

y_test_arr = np.asarray(y_test).ravel()
lims = np.array([-5, 105])
major_ticks = [0, 20, 40, 60, 80, 100]

#Figure
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300,

    sharex=False,
    sharey=False
)

axes = axes.ravel()

for idx, result in enumerate(pred_results):

    if idx >= len(axes):
        break

    ax = axes[idx]

    y_pred = np.asarray(result['y_pred']).ravel()
    model_name = result['model_name']
    r2 = result['r2']
    rmse = result['rmse']

    panel_label = string.ascii_lowercase[idx]

    #Parity plot
    ax.scatter(
        y_test_arr,
        y_pred,
        color='#56B4E9',
        edgecolors='white',
        linewidths=0.35,
        s=15,
        alpha=0.85,
        zorder=3
    )

    ax.plot(
        lims,
        lims,
        color='black',
        linewidth=0.9,
        linestyle='--',
        zorder=2
    )

    ax.fill_between(
        lims,
        lims - 10,
        lims + 10,
        color='gray',
        alpha=0.12,
        linewidth=0,
        zorder=1
    )

    metric_text = (f'$R^2$ = {r2:.4f}\n'
                   f'RMSE = {rmse:.2f}')
    ax.text(
        0.045,
        0.955,
        metric_text,
        transform=ax.transAxes,
        fontsize=FS_TEXT,
        ha='left',
        va='top',
        linespacing=1.15,
        bbox={
            'boxstyle': 'round,pad=0.28',
            'facecolor': 'white',
            'edgecolor': '#CCCCCC',
            'linewidth': 0.5,
            'alpha': 0.9
        },
        zorder=5
    )

    ax.set_title(
    '',
    loc='center',
    fontsize=FS_TITLE,
    fontweight='bold',
    pad=3)

    # Panel label: subplot 왼쪽 상단
    ax.text(
        -0.02,
        1.035,
        panel_label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        ha='left',
        va='bottom',
        clip_on=False
    )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    
    ax.set_xlabel(
    'Actual THFA yield (%)',
    fontsize=FS_LABEL,
    labelpad=2
    )
    
    ax.set_ylabel(
        'Predicted THFA yield (%)',
        fontsize=FS_LABEL,
        labelpad=2
    )

    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    ax.tick_params(
        axis='x',
        which='major',
        labelbottom=True,
        labelsize=FS_TICK,
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.tick_params(
        axis='y',
        which='major',
        labelleft=True,
        labelsize=FS_TICK,
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.set_aspect(
        'equal',
        adjustable='box'
    )

    # 네 방향이 닫힌 상자형 축
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.set_axisbelow(True)

for idx in range(len(pred_results), len(axes)):
    axes[idx].set_visible(False)

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.075,
    top=0.965,
    wspace=0.22,
    hspace=0.30
)

# plt.savefig('./figure_supp_2.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_2.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 3 (Operating time stratified SHAP summary plot)

model = XGBRegressor(random_state=SEED, n_jobs=-1)
model.load_model(f'./hyperparameter_tuning/output/xgb_model_seed_{SEED}.json')

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 165 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 6.0
FS_FEATURE = 5.2
FS_CBAR = 6.0
FS_SAMPLE = 5.2

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_FEATURE,

    'xtick.major.size': 2.5,
    'ytick.major.size': 0,

    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

rename_dict = {'Operating_temp': 'Operating temperature (°C)',
               'Operating_time': 'Operating time (h)',
               'Operating_pressure': 'Operating pressure (bar)',
               'Furfural (mg)': 'Furfural amount (mg)',
               'Active metal_Ni': 'Ni',
               'Substrate concentration (mg/ml)': 'Substrate conc. (mg/ml)',
               'Substrate to metal ratio (mmol/mmol)': 'Substrate-to-metal ratio',
               'Stirring rate (rpm)': 'Stirring rate (rpm)',
               'Reduction_temp': 'Reduction temperature (°C)',
               'Reduction_time': 'Reduction time (h)',
               'Calcination_temp': 'Calcination temperature (°C)',
               'Calcination_time': 'Calcination time (h)',
               'ethanol': 'Ethanol',
               'water': 'Water',
               '2-propanol': '2-propanol'}

time_conditions = [('Operating time < 1 h', X_train['Operating_time'] < 1),
                   ('Operating time 1–3 h', ((X_train['Operating_time'] >= 1) & (X_train['Operating_time'] < 3))),
                   ('Operating time 3–6 h', ((X_train['Operating_time'] >= 3) & (X_train['Operating_time'] <= 6))),
                   ('Operating time > 6 h', X_train['Operating_time'] > 6)]

#SHAP summary plot
explainer = shap.TreeExplainer(model)
shap_results = []

for title, condition in time_conditions:

    subset_model = X_train.loc[condition].copy()
    subset_display = subset_model.rename(
        columns=rename_dict
    )

    if len(subset_model) == 0:
        explanation = None

    else:
        sv = np.asarray(explainer.shap_values(subset_model))
        explanation = shap.Explanation(values=sv, data=subset_display.to_numpy(), feature_names=subset_display.columns.tolist())
        
    shap_results.append({'title': title,
                         'explanation': explanation,
                         'n_samples': len(subset_model)})

valid_values = [
    result['explanation'].values
    for result in shap_results
    if result['explanation'] is not None
]

if valid_values:

    global_max_abs = max(
        np.nanmax(np.abs(values))
        for values in valid_values
    )
    shap_limit = max(20, int(np.ceil(global_max_abs / 20) * 20))

else:
    shap_limit = 60


shap_ticks = np.arange(-shap_limit, shap_limit + 1, 20)

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

axes = axes.ravel()

fig.subplots_adjust(left=0.205, right=0.895, bottom=0.085, top=0.955, wspace=0.55, hspace=0.30)

for idx, (ax, result) in enumerate(zip(axes, shap_results)):

    panel_label = string.ascii_lowercase[idx]
    title = result['title']
    explanation = result['explanation']
    n_samples = result['n_samples']

    if explanation is None:

        ax.text(
            0.5,
            0.5,
            'No samples',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=FS_LABEL
        )

    else:
        shap.plots.beeswarm(
            explanation,
            max_display=15,
            ax=ax,
            show=False,
            color_bar=False,
            color=shap.plots.colors.red_blue,
            s=9,
            plot_size=None,
            group_remaining_features=False)   
        
        for line in list(ax.lines):
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)

            is_horizontal = (
                ydata.size >= 2
                and np.allclose(ydata, ydata[0])
                and not np.allclose(xdata, xdata[0])
            )
        
            if is_horizontal:
                line.remove()

    ax.set_xlim(
        -shap_limit,
        shap_limit
    )

    ax.set_xticks(shap_ticks)

    ax.set_xlabel(
        'SHAP value',
        fontsize=FS_LABEL,
        labelpad=2
    )

    ax.set_ylabel('')

    ax.tick_params(
        axis='x',
        which='major',
        labelsize=FS_TICK,
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    ax.tick_params(
        axis='y',
        which='major',
        labelsize=FS_FEATURE,
        length=0,
        pad=2
    )

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(FS_FEATURE)
        tick_label.set_fontweight('normal')

    ax.set_title(title, loc='center', fontsize=FS_TITLE, fontweight='bold', pad=3)
    ax.text(
        -0.055,
        1.025,
        panel_label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        ha='left',
        va='bottom',
        clip_on=False
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

shap_cmap = shap.plots.colors.red_blue
scalar_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=shap_cmap)
scalar_mappable.set_array([])

cbar_ax = fig.add_axes([
    0.925,   # left
    0.235,   # bottom
    0.014,   # width
    0.53     # height
])

cbar = fig.colorbar(
    scalar_mappable,
    cax=cbar_ax
)

cbar.set_ticks([0, 1])
cbar.set_ticklabels([
    'Low',
    'High'
])

cbar.set_label(
    'Feature value',
    fontsize=FS_CBAR,
    labelpad=3
)

cbar.ax.tick_params(
    labelsize=FS_FEATURE,
    direction='out',
    width=0.6,
    length=2
)

cbar.outline.set_linewidth(0.6)
# plt.savefig('./figure_supp_3.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_3.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 4 (Absolute SHAP value / category-level SHAP value)

MM_TO_INCH = 1 / 25.4

# Double-column figure
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 95 * MM_TO_INCH

# Nature figure typography
FS_PANEL = 8.0       # panel labels: a, b
FS_TITLE = 7.0
FS_AXIS = 6.5
FS_TICK = 5.5
FS_FEATURE = 5.5
FS_VALUE = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_AXIS,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_FEATURE,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # Editable text in vector figures
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

rename_dict = {'Operating_temp': 'Operating temperature (°C)',
               'Operating_time': 'Operating time (h)',
               'Operating_pressure': 'Operating pressure (bar)',
               'Furfural (mg)': 'Furfural amount (mg)',
               'Active metal_Ni': 'Ni',
               'Substrate concentration (mg/ml)': 'Substrate conc. (mg/ml)',
               'Substrate to metal ratio (mmol/mmol)': 'Substrate-to-metal ratio',
               'Stirring rate (rpm)': 'Stirring rate (rpm)',
               'Reduction_temp': 'Reduction temperature (°C)',
               'Reduction_time': 'Reduction time (h)',
               'Calcination_temp': 'Calcination temperature (°C)',
               'Calcination_time': 'Calcination time (h)',
               'ethanol': 'Ethanol',
               'water': 'Water',
               '2-propanol': '2-propanol'}

#SHAP summary plot
explainer = shap.TreeExplainer(model)
shap_values = np.asarray(explainer.shap_values(X_train))

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature_original': X_train.columns,
    'feature': [
        rename_dict.get(name, name)
        for name in X_train.columns
    ],
    'mean_abs_shap': mean_abs_shap
}).sort_values(
    'mean_abs_shap',
    ascending=False
)

active_metal_set = set(active_metal)
support_set = set(cat_support)
precursor_set = set(precursor)
preparation_set = set(preparation)
solvent_set = set(solvent)

preparation_condition = {'Calcination_temp', 'Calcination_time', 'Reduction_temp', 'Reduction_time'} | preparation_set
reaction_condition = {'Furfural (mg)', 'Catalyst amount (mg)', 'Operating_temp', 'Operating_pressure',
                      'Operating_time', 'Stirring rate (rpm)', 'Substrate to metal ratio (mmol/mmol)',
                      'Substrate concentration (mg/ml)'} | solvent_set

def categorize_feature(name):

    if name in active_metal_set:
        return 'Active metal'

    if name in support_set:
        return 'Catalyst support'

    if name in precursor_set:
        return 'Metal precursor'

    if name in preparation_condition:
        return 'Preparation method'

    if name in reaction_condition:
        return 'Reaction conditions'

    return 'Other'


shap_df['Category'] = (shap_df['feature_original'].apply(categorize_feature))

category_sums = (shap_df.groupby('Category', observed=True)['mean_abs_shap'].sum())
category_percentages = (category_sums / category_sums.sum() * 100)
category_percentages = (category_percentages.sort_values(ascending=True))

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.20, 1.00], wspace=0.40)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

#mean absolute SHAP value
top_n = 15
top_shap = (shap_df.head(top_n).sort_values('mean_abs_shap', ascending=True))

ax1.barh(
    y=top_shap['feature'],
    width=top_shap['mean_abs_shap'],
    height=0.72,
    color='#4C78A8',
    edgecolor='black',
    linewidth=0.30,
    zorder=2
)

ax1.set_xlabel('Mean absolute SHAP value', fontsize=FS_AXIS, labelpad=3)
ax1.set_ylabel('')

ax1.tick_params(axis='x', labelsize=FS_TICK, direction='out', width=0.6, length=2.5, pad=2)
ax1.tick_params(axis='y', labelsize=FS_FEATURE, direction='out', width=0.6, length=0, pad=2)
ax1.set_xlim(0,top_shap['mean_abs_shap'].max() * 1.06)
ax1.grid(False)


#category level plot
category_colors = {
    'Reaction conditions': '#0072B2',
    'Active metal': '#E69F00',
    'Catalyst support': '#009E73',
    'Preparation method': '#CC79A7',
    'Metal precursor': '#D55E00',
}

bar_colors = [category_colors.get(category, '#999999') for category in category_percentages.index]
bars = ax2.barh(y=category_percentages.index, width=category_percentages.values, height=0.62,
                color=bar_colors, edgecolor='black', linewidth=0.30, zorder=2)
ax2.set_xlabel('Category-level SHAP contribution', fontsize=FS_AXIS, labelpad=3)
ax2.set_ylabel('')
ax2.tick_params(axis='x', labelsize=FS_TICK, direction='out',
                width=0.6, length=2.5, pad=2)
ax2.tick_params(axis='y', labelsize=FS_FEATURE, direction='out',
                width=0.6, length=0, pad=2)

category_max = category_percentages.max()
ax2.set_xlim(0, np.ceil((category_max + 8) / 10) * 10)

for bar, percentage in zip(bars, category_percentages.values):
    ax2.text(percentage + 0.8, bar.get_y() + bar.get_height() / 2,
             f'{percentage:.1f}%', ha='left', va='center',
             fontsize=FS_VALUE, color='black')
ax2.grid(False)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.set_axisbelow(True)

ax1.text(-0.03, 1.035, 'a', transform=ax1.transAxes, fontsize=FS_PANEL,
         fontweight='bold', fontstyle='normal', ha='left', va='bottom',
         color='black', clip_on=False)
ax2.text(-0.03, 1.035, 'b', transform=ax2.transAxes, fontsize=FS_PANEL,
         fontweight='bold', fontstyle='normal', ha='left', va='bottom',
         color='black', clip_on=False)

fig.subplots_adjust(left=0.190, right=0.980, bottom=0.155, top=0.930, wspace=0.58)

# plt.savefig('./figure_supp_4.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_4.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 5-6 (covariate / boostrap calculation (Calculation))
X_train_float = X_train.astype(float) 
noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']

Pd_only_df = X_train_float[(X_train_float['Pd'] > 0) & ((X_train_float[active_metal] > 0).sum(axis=1) == 1)].copy()

condition_1 = X_train_float['Ni'] > 0
condition_2 = X_train_float[noble_metal].sum(axis=1) == 0

Ni_only_df = X_train_float[condition_1 & condition_2].copy()

X_train_float = X_train.astype(float)

noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']
noble_data = X_train_float[X_train_float[noble_metal].sum(axis=1) > 0]
non_noble = X_train_float[(X_train_float[noble_metal].sum(axis=1) == 0)]

family_df = pd.concat([Pd_only_df.assign(_family=1), Ni_only_df.assign(_family=0)], axis=0,ignore_index=True)
model_columns = list(X_train_float.columns)

metal_precursor_columns = precursor
metal_derived_columns = ['Substrate to metal ratio (mmol/mmol)']

exclude_from_balance = sorted(set(active_metal) | set(metal_precursor_columns) | set(metal_derived_columns))

def build_propensity_model(random_state=23, C=1.0):
    return Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median', add_indicator=True)),
               ('scaler', StandardScaler()),
               ('logistic', LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=5000, random_state=random_state))
               ])

def estimate_overlap_weights(
    data,
    balance_columns,
    family_column='_family',
    random_state=23,
    C=1.0,
    propensity_clip=1e-4):

    X_cov = (data.loc[:, balance_columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan))
    y_family = (data[family_column].astype(int).to_numpy())

    all_missing_columns = [column for column in X_cov.columns if X_cov[column].notna().sum() == 0]

    if all_missing_columns:
        X_cov = X_cov.drop(columns=all_missing_columns)

    propensity_model = build_propensity_model(random_state=random_state, C=C)
    propensity_model.fit(X_cov, y_family)
    propensity_score = (propensity_model.predict_proba(X_cov)[:, 1])
    propensity_score = np.clip(propensity_score, propensity_clip, 1.0 - propensity_clip)
    overlap_weight = np.where(y_family == 1, 1.0 - propensity_score, propensity_score)

    return {
        'propensity_score': propensity_score,
        'overlap_weight': overlap_weight,
        'propensity_model': propensity_model,
        'balance_columns_used': list(X_cov.columns),
        'all_missing_columns': all_missing_columns
    }

def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    denominator = np.sum(weights ** 2)
    if denominator <= 0:
        return np.nan

    return (np.sum(weights) ** 2 / denominator)

def weighted_mean_and_variance(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (np.isfinite(values) & np.isfinite(weights) & (weights >= 0))

    values = values[valid]
    weights = weights[valid]
    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        return np.nan, np.nan

    mean = np.sum(weights * values) / weight_sum
    variance = np.sum(weights * (values - mean) ** 2) / weight_sum

    return mean, variance


def standardized_mean_difference(values, family, weights=None):

    values = np.asarray(values, dtype=float)
    family = np.asarray(family, dtype=int)

    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    pd_mask = family == 1
    ni_mask = family == 0

    pd_mean, pd_var = weighted_mean_and_variance(values[pd_mask], weights[pd_mask])
    ni_mean, ni_var = weighted_mean_and_variance(values[ni_mask], weights[ni_mask])
    pooled_sd = np.sqrt(0.5 * (pd_var + ni_var))

    if not np.isfinite(pooled_sd):
        return np.nan

    if pooled_sd < 1e-12:
        if np.isclose(pd_mean, ni_mean):
            return 0.0

        return np.inf * np.sign(pd_mean - ni_mean)

    return (pd_mean - ni_mean) / pooled_sd

def calculate_balance_table(data, balance_columns, overlap_weights, family_column='_family'):
    family = (data[family_column].astype(int).to_numpy())
    rows = []

    for column in balance_columns:
        raw_values = (pd.to_numeric(data[column], errors='coerce').replace([np.inf, -np.inf], np.nan))

        if raw_values.notna().sum() == 0:
            continue

        missing_indicator = (
            raw_values.isna()
            .astype(float)
            .to_numpy()
        )

        median_value = raw_values.median()

        imputed_values = (
            raw_values
            .fillna(median_value)
            .to_numpy(dtype=float)
        )

        # Balance of median-imputed covariate
        smd_before = standardized_mean_difference(
            values=imputed_values,
            family=family,
            weights=None
        )

        smd_after = standardized_mean_difference(
            values=imputed_values,
            family=family,
            weights=overlap_weights
        )

        # Balance of missingness itself
        missing_smd_before = standardized_mean_difference(
            values=missing_indicator,
            family=family,
            weights=None
        )

        missing_smd_after = standardized_mean_difference(
            values=missing_indicator,
            family=family,
            weights=overlap_weights
        )

        rows.append({
            'Covariate': column,

            'SMD_before': smd_before,
            'SMD_after': smd_after,

            'Abs_SMD_before': abs(smd_before),
            'Abs_SMD_after': abs(smd_after),

            'Missing_SMD_before': missing_smd_before,
            'Missing_SMD_after': missing_smd_after,

            'Abs_Missing_SMD_before': abs(
                missing_smd_before
            ),
            'Abs_Missing_SMD_after': abs(
                missing_smd_after
            ),

            'Missing_fraction': (
                missing_indicator.mean()
            )
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            'Abs_SMD_after',
            ascending=False
        )
        .reset_index(drop=True)
    )

def make_common_grid(pd_data, ni_data, feature, n_grid=100, quantile_range=(0.0, 1.0), hard_limit=None):
    q_low, q_high = quantile_range

    pd_values = (
        pd_data[feature]
        .astype(float)
        .dropna()
        .to_numpy()
    )

    ni_values = (
        ni_data[feature]
        .astype(float)
        .dropna()
        .to_numpy()
    )

    lower = max(
        np.quantile(pd_values, q_low),
        np.quantile(ni_values, q_low)
    )

    upper = min(
        np.quantile(pd_values, q_high),
        np.quantile(ni_values, q_high)
    )

    if hard_limit is not None:
        lower = max(
            lower,
            hard_limit[0]
        )

        upper = min(
            upper,
            hard_limit[1]
        )

    return np.linspace(
        lower,
        upper,
        n_grid
    )


def build_prediction_matrix(estimator, X_model, feature, grid):

    prediction_matrix = np.empty((len(X_model), len(grid)), dtype=float)
    X_modified = X_model.copy()

    for grid_idx, feature_value in enumerate(grid):
        X_modified.loc[:, feature] = float(
            feature_value
        )

        prediction_matrix[:, grid_idx] = (
            np.asarray(
                estimator.predict(X_modified),
                dtype=float
            ).reshape(-1)
        )

    return prediction_matrix


def calculate_weighted_family_curve(prediction_matrix, family, weights, target_family):
    family = np.asarray(family, dtype=int)
    weights = np.asarray(weights, dtype=float)
    mask = family == target_family

    return np.average(
        prediction_matrix[mask, :],
        axis=0,
        weights=weights[mask]
    )

def run_overlap_weighted_family_pdp(
    estimator,
    family_data,
    model_columns,
    features,
    x_limits=None,
    exclude_balance_columns=None,
    n_grid=100,
    quantile_range=(0.0, 1.0),
    n_bootstrap=500,
    confidence_level=0.95,
    propensity_C=1.0,
    random_state=23
):

    if x_limits is None:
        x_limits = {}

    if exclude_balance_columns is None:
        exclude_balance_columns = []

    rng = np.random.default_rng(random_state)
    family_data = family_data.copy().reset_index(drop=True)
    family = family_data['_family'].astype(int).to_numpy()
    pd_indices = np.flatnonzero(family == 1)
    ni_indices = np.flatnonzero(family == 0)

    X_model = family_data[model_columns].astype(float)
    results = {}
    diagnostic_rows = []

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    for feature_idx, feature in enumerate(features):
        print(
            f'Processing {feature} '
            f'({feature_idx + 1}/{len(features)})'
        )

        excluded = (set(exclude_balance_columns) | {feature})
        balance_columns = [column for column in model_columns
                           if (column not in excluded and family_data[column].nunique(dropna=False) > 1)]

        grid = make_common_grid(
            pd_data=family_data.loc[family_data['_family'] == 1],
            ni_data=family_data.loc[family_data['_family'] == 0],
            feature=feature, n_grid=n_grid,
            quantile_range=quantile_range,
            hard_limit=x_limits.get(feature, None)
        )

        prediction_matrix = build_prediction_matrix(
            estimator=estimator, X_model=X_model, feature=feature, grid=grid)

        weight_result = estimate_overlap_weights(
            data=family_data,
            balance_columns=balance_columns,
            family_column='_family',
            random_state=random_state + feature_idx,
            C=propensity_C
        )
        
        balance_columns = weight_result[
            'balance_columns_used'
        ]
        overlap_weights = weight_result[
            'overlap_weight'
        ]

        propensity_scores = weight_result[
            'propensity_score'
        ]

        pd_curve = calculate_weighted_family_curve(
            prediction_matrix=prediction_matrix,
            family=family,
            weights=overlap_weights,
            target_family=1
        )

        ni_curve = calculate_weighted_family_curve(
            prediction_matrix=prediction_matrix,
            family=family,
            weights=overlap_weights,
            target_family=0
        )

        difference_curve = (
            ni_curve - pd_curve
        )
        
        #Bootstrap
        pd_bootstrap_curves = []
        ni_bootstrap_curves = []

        for bootstrap_idx in range(n_bootstrap):
            sampled_pd_indices = rng.choice(
                pd_indices,
                size=len(pd_indices),
                replace=True
            )

            sampled_ni_indices = rng.choice(
                ni_indices,
                size=len(ni_indices),
                replace=True
            )

            bootstrap_indices = np.concatenate(
                [
                    sampled_pd_indices,
                    sampled_ni_indices
                ]
            )

            bootstrap_data = (
                family_data.iloc[
                    bootstrap_indices
                ]
                .reset_index(drop=True)
            )

            bootstrap_family = bootstrap_data[
                '_family'
            ].astype(int).to_numpy()

            try:
                bootstrap_weight_result = estimate_overlap_weights(
                    data=bootstrap_data,
                    balance_columns=balance_columns,
                    family_column='_family',
                    random_state=(
                        random_state
                        + feature_idx * 100_000
                        + bootstrap_idx
                    ),
                    C=propensity_C
                )
                
                bootstrap_weights = bootstrap_weight_result[
                    'overlap_weight'
                ]

                bootstrap_prediction_matrix = (
                    prediction_matrix[
                        bootstrap_indices,
                        :
                    ]
                )

                pd_bootstrap_curve = (
                    calculate_weighted_family_curve(
                        prediction_matrix=bootstrap_prediction_matrix,
                        family=bootstrap_family,
                        weights=bootstrap_weights,
                        target_family=1
                    )
                )

                ni_bootstrap_curve = (
                    calculate_weighted_family_curve(
                        prediction_matrix=bootstrap_prediction_matrix,
                        family=bootstrap_family,
                        weights=bootstrap_weights,
                        target_family=0
                    )
                )

                pd_bootstrap_curves.append(
                    pd_bootstrap_curve
                )

                ni_bootstrap_curves.append(
                    ni_bootstrap_curve
                )

            except (
                ValueError,
                FloatingPointError
            ):
                continue

        pd_bootstrap_curves = np.asarray(
            pd_bootstrap_curves,
            dtype=float
        )

        ni_bootstrap_curves = np.asarray(
            ni_bootstrap_curves,
            dtype=float
        )

        difference_bootstrap_curves = (
            ni_bootstrap_curves
            - pd_bootstrap_curves
        )

        pd_lower = np.quantile(
            pd_bootstrap_curves,
            lower_quantile,
            axis=0
        )

        pd_upper = np.quantile(
            pd_bootstrap_curves,
            upper_quantile,
            axis=0
        )

        ni_lower = np.quantile(
            ni_bootstrap_curves,
            lower_quantile,
            axis=0
        )

        ni_upper = np.quantile(
            ni_bootstrap_curves,
            upper_quantile,
            axis=0
        )

        difference_lower = np.quantile(
            difference_bootstrap_curves,
            lower_quantile,
            axis=0
        )

        difference_upper = np.quantile(
            difference_bootstrap_curves,
            upper_quantile,
            axis=0
        )

        probability_ni_above_pd = np.mean(
            difference_bootstrap_curves > 0,
            axis=0
        )

        balance_table = calculate_balance_table(
            data=family_data,
            balance_columns=balance_columns,
            overlap_weights=overlap_weights,
            family_column='_family'
        )

        pd_weights = overlap_weights[
            family == 1
        ]

        ni_weights = overlap_weights[
            family == 0
        ]

        diagnostic_rows.append(
            {
                'Feature': feature,
                'Grid_min': grid.min(),
                'Grid_max': grid.max(),
                'Number_of_balance_covariates': len(
                    balance_columns
                ),
                'Mean_abs_SMD_before': (
                    balance_table[
                        'Abs_SMD_before'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).mean()
                ),
                'Mean_abs_SMD_after': (
                    balance_table[
                        'Abs_SMD_after'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).mean()
                ),
                'Max_abs_SMD_before': (
                    balance_table[
                        'Abs_SMD_before'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).max()
                ),
                'Max_abs_SMD_after': (
                    balance_table[
                        'Abs_SMD_after'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).max()
                ),
                'Pd_effective_sample_size': (
                    effective_sample_size(
                        pd_weights
                    )
                ),
                'Ni_effective_sample_size': (
                    effective_sample_size(
                        ni_weights
                    )
                ),
                'Successful_bootstraps': len(
                    pd_bootstrap_curves
                ),
                'Propensity_min': propensity_scores.min(),
                'Propensity_max': propensity_scores.max()
            }
        )

        results[feature] = {
            'grid': grid,

            'pd_curve': pd_curve,
            'pd_lower': pd_lower,
            'pd_upper': pd_upper,

            'ni_curve': ni_curve,
            'ni_lower': ni_lower,
            'ni_upper': ni_upper,

            'difference_curve': difference_curve,
            'difference_lower': difference_lower,
            'difference_upper': difference_upper,

            'probability_ni_above_pd': (
                probability_ni_above_pd
            ),

            'propensity_score': propensity_scores,
            'overlap_weight': overlap_weights,

            'balance_columns': balance_columns,
            'balance_table': balance_table,

            'pd_bootstrap_curves': pd_bootstrap_curves,
            'ni_bootstrap_curves': ni_bootstrap_curves
        }

    diagnostic_summary = pd.DataFrame(
        diagnostic_rows
    )

    return results, diagnostic_summary

#%% Supplementary figures - Fig. 5
features = [
    'Operating_temp',
    'Operating_pressure',
    'Operating_time'
]

x_labels = [
    'Operating temperature (°C)',
    'Operating pressure (bar)',
    'Operating time (h)'
]

x_lims = {
    'Operating_temp': (50, 200),
    'Operating_pressure': (0, 40),
    'Operating_time': (0, 10)
}

x_ticks = {
    'Operating_temp': [50, 100, 150, 200],
    'Operating_pressure': [0, 10, 20, 30, 40],
    'Operating_time': [0, 2, 4, 6, 8, 10]
}

balanced_pdp_results, balance_summary = (
    run_overlap_weighted_family_pdp(
        estimator=model,
        family_data=family_df,
        model_columns=model_columns,
        features=features,
        x_limits=x_lims,
        exclude_balance_columns=exclude_from_balance,
        n_grid=100,
        quantile_range=(0.0, 1.0),
        n_bootstrap=500,

        confidence_level=0.95,
        propensity_C=1.0,
        random_state=23
    )
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

n_pd = int((family_df['_family'] == 1).sum())
n_ni = int((family_df['_family'] == 0).sum())

balance_check = balance_summary.copy()

balance_check['Pd_original_n'] = n_pd
balance_check['Ni_original_n'] = n_ni

balance_check['Pd_ESS_ratio'] = (
    balance_check['Pd_effective_sample_size']
    / n_pd
)

balance_check['Ni_ESS_ratio'] = (
    balance_check['Ni_effective_sample_size']
    / n_ni
)

diagnostic_columns = [
    'Feature',
    'Grid_min',
    'Grid_max',
    'Number_of_balance_covariates',
    'Mean_abs_SMD_before',
    'Mean_abs_SMD_after',
    'Max_abs_SMD_before',
    'Max_abs_SMD_after',
    'Pd_effective_sample_size',
    'Pd_ESS_ratio',
    'Ni_effective_sample_size',
    'Ni_ESS_ratio',
    'Successful_bootstraps',
    'Propensity_min',
    'Propensity_max'
]

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 102 * MM_TO_INCH

COLOR_PD = '#D95F5F'
COLOR_NI = '#4C78A8'

FS_AXIS = 7.5
FS_TICK = 7.0
FS_TITLE = 7.5
FS_LEGEND = 7.0
FS_PANEL = 8.5

LW_AXIS = 0.65
LW_LINE = 1.0

rcParams.update({
    'font.family': 'Arial',
    'font.size': FS_AXIS,

    'axes.linewidth': LW_AXIS,
    'axes.labelsize': FS_AXIS,
    'axes.titlesize': FS_TITLE,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'legend.fontsize': FS_LEGEND,

    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

pd_label = 'Pd-only catalysts'
ni_label = 'Ni-based non-noble metal catalysts'

bin_edges = np.linspace(0, 1, 31)

#Figure
fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))

for idx, feature in enumerate(features):
    result = balanced_pdp_results[feature]
    propensity = np.asarray(result['propensity_score'], dtype=float)
    weights = np.asarray(result['overlap_weight'], dtype=float)
    family = (family_df['_family'].astype(int).to_numpy())

    valid = (np.isfinite(propensity) & np.isfinite(weights))
    propensity_valid = propensity[valid]
    weights_valid = weights[valid]
    family_valid = family[valid]
    pd_mask = family_valid == 1
    ni_mask = family_valid == 0

    ax = axes[0, idx]
    ax.hist(
        propensity_valid[pd_mask],
        bins=bin_edges,
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_PD,
        label=pd_label
    )
    ax.hist(
        propensity_valid[ni_mask],
        bins=bin_edges,
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_NI,
        label=ni_label
    )
    
    ax.set_xlim(0, 1)
    
    if idx == 0:
        ax.set_ylabel('Density', labelpad=3)
    else:
        ax.set_ylabel('')

    ax = axes[1, idx]
    ax.hist(
        propensity_valid[pd_mask],
        bins=bin_edges,
        weights=weights_valid[pd_mask],
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_PD,
        label=pd_label
    )
    ax.hist(
        propensity_valid[ni_mask],
        bins=bin_edges,
        weights=weights_valid[ni_mask],
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_NI,
        label=ni_label
    )

    ax.set_xlim(0, 1)

    ax.set_xlabel(
        'Propensity score for\nmonometallic Pd catalysts',
        labelpad=3
    )

    if idx == 0:
        ax.set_ylabel('Density', labelpad=3)
    else:
        ax.set_ylabel('')

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

for panel_idx, ax in enumerate(axes.ravel()):

    ax.text(
        -0.16,
        1.06,
        panel_labels[panel_idx],
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        va='top',
        ha='left'
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(LW_AXIS)
    ax.spines['bottom'].set_linewidth(LW_AXIS)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.grid(False)

legend_handles = [
    Line2D(
        [0], [0],
        color=COLOR_PD,
        linewidth=2.0,
        label='Monometallic Pd catalysts'
    ),
    Line2D(
        [0], [0],
        color=COLOR_NI,
        linewidth=2.0,
        label='Ni-based non-noble catalysts'
    )
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.53, 0.95),
    ncol=2,
    frameon=False,
    handlelength=2.5,
    columnspacing=1.8,
    handletextpad=0.6
)

fig.subplots_adjust(
    left=0.095,
    right=0.99,
    bottom=0.13,
    top=0.85,
    wspace=0.27,
    hspace=0.34
)

# plt.savefig('./figure_supp_5.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_5.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 6
MM_TO_INCH = 1 / 25.4

COLOR_PD = '#D95F5F'
COLOR_NI = '#4C78A8'

FS_AXIS = 7.5
FS_TICK = 7.0
FS_LEGEND = 7.2
FS_PANEL = 8.5

LW_AXIS = 0.65
LW_CURVE = 1.35
LW_RUG = 0.45

rcParams.update({
    'font.family': 'Arial',
    'font.size': FS_AXIS,

    'axes.linewidth': LW_AXIS,
    'axes.labelsize': FS_AXIS,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'legend.fontsize': FS_LEGEND,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    'savefig.transparent': False
})

y_lim = (-12, 62)
y_ticks = [0, 20, 40, 60]

panel_labels = ['a', 'b', 'c']

features = [
    'Operating_temp',
    'Operating_pressure',
    'Operating_time'
]

x_ticks = [
    [50, 100, 150, 200],
    [0, 10, 20, 30, 40],
    [0, 2, 4, 6, 8, 10]
]

x_lims = [
    (50, 200),
    (0, 40),
    (0, 10)
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        180 * MM_TO_INCH,
        62 * MM_TO_INCH
    ),
    sharey=True
)

for idx, (ax, feature, x_label) in enumerate(
    zip(axes, features, x_labels)
):
    result = balanced_pdp_results[feature]

    grid = np.asarray(
        result['grid'],
        dtype=float
    )

    pd_curve = np.asarray(
        result['pd_curve'],
        dtype=float
    )

    pd_lower = np.asarray(
        result['pd_lower'],
        dtype=float
    )

    pd_upper = np.asarray(
        result['pd_upper'],
        dtype=float
    )

    ni_curve = np.asarray(
        result['ni_curve'],
        dtype=float
    )

    ni_lower = np.asarray(
        result['ni_lower'],
        dtype=float
    )

    ni_upper = np.asarray(
        result['ni_upper'],
        dtype=float
    )


    valid_pd = (
        np.isfinite(grid)
        & np.isfinite(pd_curve)
        & np.isfinite(pd_lower)
        & np.isfinite(pd_upper)
    )

    valid_ni = (
        np.isfinite(grid)
        & np.isfinite(ni_curve)
        & np.isfinite(ni_lower)
        & np.isfinite(ni_upper)
    )


    ax.fill_between(
        grid[valid_pd],
        pd_lower[valid_pd],
        pd_upper[valid_pd],
        color=COLOR_PD,
        alpha=0.18,
        linewidth=0,
        zorder=1
    )

    ax.fill_between(
        grid[valid_ni],
        ni_lower[valid_ni],
        ni_upper[valid_ni],
        color=COLOR_NI,
        alpha=0.18,
        linewidth=0,
        zorder=1
    )


    ax.plot(
        grid[valid_pd],
        pd_curve[valid_pd],
        color=COLOR_PD,
        linewidth=LW_CURVE,
        solid_capstyle='round',
        zorder=3
    )

    ax.plot(
        grid[valid_ni],
        ni_curve[valid_ni],
        color=COLOR_NI,
        linewidth=LW_CURVE,
        solid_capstyle='round',
        zorder=3
    )


    pd_rug_values = (
        pd.to_numeric(
            Pd_only_df[feature],
            errors='coerce'
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    ni_rug_values = (
        pd.to_numeric(
            Ni_only_df[feature],
            errors='coerce'
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    # Pd rugs
    ax.plot(
        pd_rug_values,
        np.full(
            pd_rug_values.shape,
            0.025
        ),
        linestyle='None',
        marker='|',
        markersize=3.0,
        markeredgewidth=LW_RUG,
        color=COLOR_PD,
        alpha=0.45,
        transform=ax.get_xaxis_transform(),
        clip_on=True,
        zorder=4
    )

    # Ni rugs
    ax.plot(
        ni_rug_values,
        np.full(
            ni_rug_values.shape,
            0.055
        ),
        linestyle='None',
        marker='|',
        markersize=3.0,
        markeredgewidth=LW_RUG,
        color=COLOR_NI,
        alpha=0.45,
        transform=ax.get_xaxis_transform(),
        clip_on=True,
        zorder=4
    )


    ax.axhline(
        0,
        color='0.55',
        linewidth=0.55,
        linestyle='-',
        zorder=0
    )


    ax.set_xlim(
        x_lims[idx]
    )
    
    ax.set_ylim(
        y_lim
    )
    
    ax.set_xticks(
        x_ticks[idx]
    )
    
    ax.set_yticks(
        y_ticks
    )
    
    ax.set_xlabel(
        x_label,
        labelpad=3
    )

    if idx == 0:
        ax.set_ylabel(
            'Overlap-weighted mean predicted\nTHFA yield (%)',
            labelpad=4
        )
    else:
        ax.set_ylabel('')

    ax.text(
        -0.17,
        1.04,
        panel_labels[idx],
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        va='bottom',
        ha='left'
    )


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(LW_AXIS)
    ax.spines['bottom'].set_linewidth(LW_AXIS)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.grid(False)

legend_handles = [
    Line2D(
        [0], [0],
        color=COLOR_PD,
        linewidth=2.0,
        solid_capstyle='round',
        label='Monometallic Pd catalysts'
    ),

    Line2D(
        [0], [0],
        color=COLOR_NI,
        linewidth=2.0,
        solid_capstyle='round',
        label='Ni-based noble-metal-free catalysts'
    )
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.53, 1.015),
    ncol=2,
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=2.4,
    handletextpad=0.6,
    columnspacing=1.8,
    borderaxespad=0
)

fig.subplots_adjust(
    left=0.105,
    right=0.992,
    bottom=0.20,
    top=0.79,
    wspace=0.20
)

# plt.savefig('./figure_supp_6.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_6.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 7 (Potential space for 1X-4Ni candidates (Calculation))

with open('./dataset/molar_mass.pickle', 'rb') as f:
    molar_mass = pickle.load(f)
# ['Ca','Co','Cu','Fe','In','Ir','Ni', 'Pd','Pt', 'Re', 'Rh', 'Ru', 'Sn', 'Zn']

precursor_map = {
    'Ca': 'Ca(NO3)2',   
    'Co': 'Co(NO3)2',
    'Cu': 'Cu(NO3)2',
    'Fe': 'Fe(NO3)3', 
    'In': 'In(SO3CF3)3', 
    'Ir': 'IrCl3',
    'Ni': 'Ni(NO3)2', 
    'Pd': 'PdCl2', 
    'Pt': 'H2PtCl6', 
    'Re': 'NH4ReO4', 
    'Rh': 'RhCl3',   
    'Ru': 'RuCl3', 
    'Sn': 'SnCl4',    
    'Zn': 'Zn(NO3)2'} 

selected_support = 'Al2O3' 
selected_solvent = '2-propanol' 
solvent_amount   = 40
selected_preparation = 'wet impregnation'
calc_temp, calc_time = 500, 6
reduc_temp, reduc_time = 400, 4
oper_time =  6
stir_rate = 700
cat_amount, FF_amount = 100, 300


temp_range = np.arange(120, 201, 1)
pres_range = np.arange(10, 40.5, 0.5)

pd_rows_list = []
for t in temp_range:
    for p in pres_range:
        
        new_row = {col: 0 for col in dataset.columns[:-1]}
        
        AM_1 = 'Pd'
        precursor_1 = precursor_map[AM_1]
        
        new_row[AM_1] = 5.0
        new_row[precursor_1] = 1
        new_row[selected_support] = 95.0
        new_row[selected_preparation] = 1 
        new_row[selected_solvent] = solvent_amount
        new_row['Stirring rate (rpm)'] = stir_rate
        new_row['Catalyst amount (mg)'] = cat_amount
        new_row['Furfural (mg)'] = FF_amount
        new_row['Operating_temp'] = t
        new_row['Operating_pressure'] = p
        new_row['Operating_time'] = oper_time
        new_row['Calcination_temp'] = calc_temp
        new_row['Reduction_temp'] = reduc_temp
        new_row['Calcination_time'] = calc_time
        new_row['Reduction_time'] = reduc_time
        furfural_mmol   = float(FF_amount / molar_mass['Furfural'])
        AM1_percent =  5.0 * 0.01    
        subs_to_metal = float(furfural_mmol / (cat_amount * AM1_percent / molar_mass[AM_1]))
        subs_concentration = float(FF_amount / solvent_amount)
        new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
        new_row['Substrate concentration (mg/ml)'] = subs_concentration
        
        pd_rows_list.append(new_row)

Pd_baseline_df = pd.DataFrame(pd_rows_list)
X_features = dataset.columns.tolist()[:-1] 
y_pred = model.predict(Pd_baseline_df[X_features])
y_pred = np.clip(y_pred, 0, 100)
Pd_baseline_df['THFA_yield (%)'] = y_pred
Pd_baseline_df['Combination'] = 'Pd (5wt%)'

non_noble_target = ['Co', 'Fe', 'Cu', 'Ca', 'Zn', 'Re']
total_heatmap_data ={}

for asd in non_noble_target:
    ni_rows_list = []
    for t in temp_range:
        for p in pres_range:
            
            new_row = {col: 0 for col in dataset.columns[:-1]}
            
            AM_1, AM_2 = 'Ni', asd
            precursor_1 = precursor_map[AM_1]
            precursor_2 = precursor_map[AM_2]
            new_row[AM_1], new_row[AM_2] = 4.0, 1.0
            new_row[precursor_1], new_row[precursor_2] = 1, 1
            new_row[selected_support] = 95.0
            new_row[selected_preparation] = 1 
            new_row[selected_solvent] = solvent_amount
            new_row['Stirring rate (rpm)'] = stir_rate
            new_row['Catalyst amount (mg)'] = cat_amount
            new_row['Furfural (mg)'] = FF_amount
            new_row['Operating_temp'] = t
            new_row['Operating_pressure'] = p
            new_row['Operating_time'] = oper_time
            new_row['Calcination_temp'] = calc_temp
            new_row['Reduction_temp'] = reduc_temp
            new_row['Calcination_time'] = calc_time
            new_row['Reduction_time'] = reduc_time
            furfural_mmol   = float(FF_amount / molar_mass['Furfural'])
            AM1_percent = 4.0 * 0.01    
            AM2_percent = 1.0 * 0.01 
            subs_to_metal = float(furfural_mmol / (cat_amount * AM1_percent / molar_mass[AM_1] + cat_amount * AM2_percent / molar_mass[AM_2]))
            subs_concentration = float(FF_amount / solvent_amount)
            new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
            new_row['Substrate concentration (mg/ml)'] = subs_concentration
            
            ni_rows_list.append(new_row)
    
    Ni_X_baseline_df = pd.DataFrame(ni_rows_list)
    X_features = dataset.columns.tolist()[:-1] 
    y_pred = model.predict(Ni_X_baseline_df[X_features])
    y_pred = np.clip(y_pred, 0, 100)
    Ni_X_baseline_df['THFA_yield (%)'] = y_pred
    Ni_X_baseline_df['Combination'] = f'Ni (4wt%) / {AM_2} (1wt%)'
    
    Ni_X_baseline_df['delta_y'] = (
        Ni_X_baseline_df['THFA_yield (%)'].values
        - Pd_baseline_df['THFA_yield (%)'].values
    )
    
    heatmap_data = Ni_X_baseline_df.pivot_table(
    index='Operating_pressure',
    columns='Operating_temp',
    values='delta_y')
    
    total_heatmap_data[f'1{asd}-4Ni'] = heatmap_data

non_noble_target = ['Co', 'Fe', 'Cu', 'Ca', 'Zn', 'Re']

vmin, vmax = -70, 20
v_range = vmax - vmin
clevels = np.linspace(vmin, vmax, 200)


color_nodes = [
    (0.0, '#001529'),                        
    (( -35 - vmin) / v_range, '#1e466e'),      
    (( -10 - vmin) / v_range, '#85a5c2'),     
    ((  0 - vmin) / v_range, '#ffffff'),     
    ((  5 - vmin) / v_range, '#e6c875'),      
    (( 12 - vmin) / v_range, '#d76445'),    
    (1.0, '#9c1515')                        
]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('potential_map', color_nodes)

#%% Supplementary figures - Fig. 7 (Potential space for 1X-4Ni candidates (figure))
#Plot
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 170 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 6.5
FS_AXIS = 6.5
FS_TICK = 5.5
FS_CBAR = 6.5
FS_CBAR_TICK = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_AXIS,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

panel_labels = string.ascii_lowercase[:len(non_noble_target)]

x_ticks = [120, 140, 160, 180, 200]
y_ticks = [10, 15, 20, 25, 30, 35, 40]

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1.0, 1.0],
                      left=0.095, right=0.895, bottom=0.075, top=0.965, wspace=0.22, hspace=0.28)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1])
]

contour = None

for idx, metal in enumerate(non_noble_target):

    ax = axes[idx]

    heatmap_key = f'1{metal}-4Ni'
    heatmap_data = total_heatmap_data[heatmap_key]

    temperature = heatmap_data.columns.to_numpy(dtype=float)
    pressure = heatmap_data.index.to_numpy(dtype=float)
    ZZ = heatmap_data.to_numpy(dtype=float)
    TT, PP = np.meshgrid(temperature, pressure)
    contour = ax.contourf(TT, PP, ZZ, levels=clevels,
                          cmap=custom_cmap, vmin=vmin, vmax=vmax, extend='both', antialiased=False)

    if hasattr(contour, 'collections'):
        for collection in contour.collections:
            collection.set_edgecolor('face')
            collection.set_linewidth(0.0)
            collection.set_antialiased(False)
            collection.set_rasterized(True)

    finite_values = ZZ[np.isfinite(ZZ)]

    if (finite_values.size > 0 and np.nanmin(finite_values) < 0 and np.nanmax(finite_values) > 0):
        zero_contour = ax.contour(TT, PP, ZZ, levels=[0], colors='black',
                                  linestyles='--', linewidths=0.8, zorder=4)

        if hasattr(zero_contour, 'collections'):
            for collection in zero_contour.collections:
                collection.set_rasterized(False)

    ax.text(-0.025, 1.035, panel_labels[idx], transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom', clip_on=False)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis='both', which='major', labelsize=FS_TICK, direction='out',
                   width=0.6, length=2.5, pad=2)
    ax.tick_params(axis='x', labelbottom=True)
    ax.tick_params(axis='y', labelleft=True)

    if idx in [4, 5]:
        ax.set_xlabel('Operating temperature (°C)', fontsize=FS_AXIS, labelpad=3)
    else:
        ax.set_xlabel('')

    if idx in [0, 2, 4]:
        ax.set_ylabel('Operating pressure (bar)', fontsize=FS_AXIS, labelpad=3)
    else:
        ax.set_ylabel('')

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

for idx in range(len(non_noble_target), len(axes)):
    axes[idx].set_visible(False)

fig.canvas.draw()
right_axes = [ax for idx, ax in enumerate(axes) if (idx % 2 == 1 and ax.get_visible())]
right_edge = max(ax.get_position().x1 for ax in right_axes)
top_edge = max(ax.get_position().y1 for ax in axes if ax.get_visible())
bottom_edge = min(ax.get_position().y0 for ax in axes if ax.get_visible())

cbar_gap = 0.030
cbar_width = 0.025

cax = fig.add_axes([right_edge + cbar_gap, bottom_edge, cbar_width, top_edge - bottom_edge])
cbar = fig.colorbar(contour, cax=cax, extend='both')
cbar.set_ticks([-60, -40, -20, 0, 20])
cbar.set_label(r'Yield difference, $\Delta Y$ (%)', fontsize=FS_CBAR, labelpad=4)
cbar.ax.tick_params(axis='y', labelsize=FS_CBAR_TICK, direction='out', width=0.6, length=2.5, pad=2)
cbar.outline.set_linewidth(0.6)
# plt.savefig('./figure_supp_7.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_7.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 8 (bootstrap stability (Calculation))

B = 200
BOOTSTRAP_SEED = 20260721
MODEL_SEED = 23

# True: calculate the bootstrap ensemble again
# False: load bootstrap_ensemble_results.npz when it already exists
RECALCULATE = False

DATA_DIR = "./dataset"
MODEL_DIR = "./hyperparameter_tuning/output"

X_TRAIN_PATH = f"{DATA_DIR}/ML_dataset_final_x_train.csv"
Y_TRAIN_PATH = f"{DATA_DIR}/ML_dataset_final_y_train.csv"
X_TEST_PATH = f"{DATA_DIR}/ML_dataset_final_x_test.csv"
Y_TEST_PATH = f"{DATA_DIR}/ML_dataset_final_y_test.csv"
MOLAR_MASS_PATH = f"{DATA_DIR}/molar_mass.pickle"

PARAM_PATH = f"{MODEL_DIR}/xgb_params_seed_{MODEL_SEED}.pkl"
MODEL_PATH = f"{MODEL_DIR}/xgb_model_seed_{MODEL_SEED}.json"

RESULT_PATH = "./bootstrap_output/bootstrap_ensemble_results.npz"
SUMMARY_PATH = "./bootstrap_output/positive_region_stability_summary.csv"

# =============================================================================
# Candidate and operating conditions
# =============================================================================
candidate_metals = ["Co", "Fe", "Cu", "Ca", "Zn", "Re"]
candidate_labels = [f"1{metal}-4Ni" for metal in candidate_metals]
thresholds = np.arange(-10.0, 11.0, 1.0)

temperature_range = np.arange(120.0, 201.0, 1.0)
pressure_range = np.arange(10.0, 40.5, 0.5)

precursor_map = {
    "Ca": "Ca(NO3)2",
    "Co": "Co(NO3)2",
    "Cu": "Cu(NO3)2",
    "Fe": "Fe(NO3)3",
    "Ni": "Ni(NO3)2",
    "Pd": "PdCl2",
    "Re": "NH4ReO4",
    "Zn": "Zn(NO3)2",
}

support = "Al2O3"
solvent = "2-propanol"
preparation = "wet impregnation"

solvent_amount = 40.0
calcination_temperature = 500.0
calcination_time = 6.0
reduction_temperature = 400.0
reduction_time = 4.0
operating_time = 6.0
stirring_rate = 700.0
catalyst_amount = 100.0
furfural_amount = 300.0


# =============================================================================
# Helper functions
# =============================================================================
def build_case(feature_names, molar_mass, active_metals):
    """Create one catalyst case over the temperature-pressure grid."""
    temperature_grid, pressure_grid = np.meshgrid(
        temperature_range,
        pressure_range,
        indexing="xy",
    )

    temperatures = temperature_grid.ravel()
    pressures = pressure_grid.ravel()

    case = pd.DataFrame(
        0.0,
        index=np.arange(len(temperatures)),
        columns=feature_names,
    )

    for metal, loading in active_metals.items():
        case[metal] = loading
        case[precursor_map[metal]] = 1.0

    case[support] = 100.0 - sum(active_metals.values())
    case[preparation] = 1.0
    case[solvent] = solvent_amount

    case["Stirring rate (rpm)"] = stirring_rate
    case["Catalyst amount (mg)"] = catalyst_amount
    case["Furfural (mg)"] = furfural_amount

    case["Operating_temp"] = temperatures
    case["Operating_pressure"] = pressures
    case["Operating_time"] = operating_time

    case["Calcination_temp"] = calcination_temperature
    case["Calcination_time"] = calcination_time
    case["Reduction_temp"] = reduction_temperature
    case["Reduction_time"] = reduction_time

    furfural_mmol = furfural_amount / molar_mass["Furfural"]

    metal_mmol = sum(
        catalyst_amount
        * (loading / 100.0)
        / molar_mass[metal]
        for metal, loading in active_metals.items()
    )

    case["Substrate to metal ratio (mmol/mmol)"] = (
        furfural_mmol / metal_mmol
    )
    case["Substrate concentration (mg/ml)"] = (
        furfural_amount / solvent_amount
    )

    return case, temperatures, pressures


def predict_delta(model, pd_case, candidate_cases):
    """Calculate candidate yield minus Pd yield over the operating grid."""
    pd_prediction = np.clip(
        model.predict(pd_case),
        0.0,
        100.0,
    )

    delta = []

    for label in candidate_labels:
        candidate_prediction = np.clip(
            model.predict(candidate_cases[label]),
            0.0,
            100.0,
        )
        delta.append(candidate_prediction - pd_prediction)

    return np.asarray(delta, dtype=np.float32)


# =============================================================================
# Load data and model information
# =============================================================================
x_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]

x_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

with open(MOLAR_MASS_PATH, "rb") as file:
    molar_mass = pickle.load(file)

with open(PARAM_PATH, "rb") as file:
    xgb_params = pickle.load(file)


# =============================================================================
# Build Pd and candidate operating grids
# =============================================================================
pd_case, temperatures, pressures = build_case(
    feature_names=x_train.columns,
    molar_mass=molar_mass,
    active_metals={"Pd": 5.0},
)

candidate_cases = {}

for metal, label in zip(candidate_metals, candidate_labels):
    candidate_cases[label], _, _ = build_case(
        feature_names=x_train.columns,
        molar_mass=molar_mass,
        active_metals={
            "Ni": 4.0,
            metal: 1.0,
        },
    )

print(f"Training data: {x_train.shape}")
print(f"Test data: {x_test.shape}")
print(f"Operating grid: {len(pd_case):,} points")


# =============================================================================
# Load existing bootstrap result or calculate again
# =============================================================================
if os.path.exists(RESULT_PATH) and not RECALCULATE:
    result = np.load(RESULT_PATH)

    bootstrap_delta = result["delta_y"]
    test_r2 = result["test_r2"]
    test_rmse = result["test_rmse"]
    mean_gap = result["mean_gap"]
    fraction_by_threshold = result["fraction_by_threshold"]
    ranks_by_threshold = result["ranks_by_threshold"]
    nominal_delta = result["nominal_delta_y"]

    print(f"Loaded: {RESULT_PATH}")

else:
    # -------------------------------------------------------------------------
    # Nominal model
    # -------------------------------------------------------------------------
    nominal_model = XGBRegressor(
        random_state=MODEL_SEED,
        n_jobs=-1,
        **xgb_params,
    )
    nominal_model.load_model(MODEL_PATH)

    nominal_delta = predict_delta(
        model=nominal_model,
        pd_case=pd_case,
        candidate_cases=candidate_cases,
    )

    re_index = candidate_labels.index("1Re-4Ni")

    nominal_re_fraction = (
        np.mean(nominal_delta[re_index] >= 0.0)
        * 100.0
    )
    nominal_re_mean_gap = np.mean(
        nominal_delta[re_index]
    )

    print(
        f"Nominal Re-Ni fraction: {nominal_re_fraction:.3f}%"
    )
    print(
        f"Nominal Re-Ni mean ΔY: {nominal_re_mean_gap:.3f} pp"
    )

    # -------------------------------------------------------------------------
    # Bootstrap refits
    # -------------------------------------------------------------------------
    n_train = len(x_train)
    n_candidates = len(candidate_labels)
    n_grid = len(pd_case)

    bootstrap_delta = np.zeros(
        (B, n_candidates, n_grid),
        dtype=np.float32,
    )
    test_r2 = np.zeros(B)
    test_rmse = np.zeros(B)

    start_time = time.time()

    for bootstrap_index in range(B):
        rng = np.random.default_rng(
            BOOTSTRAP_SEED + bootstrap_index
        )

        sampled_rows = rng.integers(
            0,
            n_train,
            size=n_train,
        )

        model = XGBRegressor(
            random_state=BOOTSTRAP_SEED + bootstrap_index,
            n_jobs=-1,
            **xgb_params,
        )

        model.fit(
            x_train.iloc[sampled_rows],
            y_train.iloc[sampled_rows],
            verbose=False,
        )

        bootstrap_delta[bootstrap_index] = predict_delta(
            model=model,
            pd_case=pd_case,
            candidate_cases=candidate_cases,
        )

        test_prediction = np.clip(
            model.predict(x_test),
            0.0,
            100.0,
        )

        test_r2[bootstrap_index] = r2_score(
            y_test,
            test_prediction,
        )
        test_rmse[bootstrap_index] = np.sqrt(
            mean_squared_error(
                y_test,
                test_prediction,
            )
        )

        if (
            (bootstrap_index + 1) % 10 == 0
            or bootstrap_index + 1 == B
        ):
            elapsed_min = (time.time() - start_time) / 60.0

            print(
                f"Bootstrap {bootstrap_index + 1}/{B} | "
                f"RMSE={test_rmse[bootstrap_index]:.3f} | "
                f"elapsed={elapsed_min:.1f} min"
            )

    # -------------------------------------------------------------------------
    # Positive-region fractions and candidate rankings
    # -------------------------------------------------------------------------
    mean_gap = np.mean(
        bootstrap_delta,
        axis=2,
    )

    fraction_by_threshold = np.mean(
        bootstrap_delta[:, :, :, None]
        >= thresholds[None, None, None, :],
        axis=2,
    ) * 100.0

    ranks_by_threshold = np.zeros(
        fraction_by_threshold.shape,
        dtype=np.int16,
    )

    for bootstrap_index in range(B):
        for threshold_index in range(len(thresholds)):
            # Larger region fraction is better.
            # Mean ΔY is used only to resolve ties.
            order = np.lexsort(
                (
                    -mean_gap[bootstrap_index],
                    -fraction_by_threshold[
                        bootstrap_index,
                        :,
                        threshold_index,
                    ],
                )
            )

            ranks_by_threshold[
                bootstrap_index,
                order,
                threshold_index,
            ] = np.arange(
                1,
                len(candidate_labels) + 1,
            )

    np.savez_compressed(
        RESULT_PATH,
        delta_y=bootstrap_delta,
        test_r2=test_r2,
        test_rmse=test_rmse,
        mean_gap=mean_gap,
        fraction_by_threshold=fraction_by_threshold,
        ranks_by_threshold=ranks_by_threshold,
        thresholds=thresholds,
        candidate_labels=np.asarray(candidate_labels),
        temperatures=temperatures,
        pressures=pressures,
        nominal_delta_y=nominal_delta,
    )

    print(f"Saved: {RESULT_PATH}")


# =============================================================================
# Save a compact summary table
# =============================================================================
zero_index = np.flatnonzero(
    np.isclose(thresholds, 0.0)
)[0]

fractions_zero = fraction_by_threshold[
    :,
    :,
    zero_index,
]

ranks_zero = ranks_by_threshold[
    :,
    :,
    zero_index,
]

summary = []

for candidate_index, label in enumerate(candidate_labels):
    values = fractions_zero[:, candidate_index]

    summary.append(
        {
            "Catalyst": label,
            "Nominal fraction (%)": (
                np.mean(
                    nominal_delta[candidate_index] >= 0.0
                )
                * 100.0
            ),
            "Bootstrap median fraction (%)": np.median(values),
            "Bootstrap 2.5th fraction (%)": np.percentile(
                values,
                2.5,
            ),
            "Bootstrap 97.5th fraction (%)": np.percentile(
                values,
                97.5,
            ),
            "Non-zero region frequency (%)": np.mean(
                values > 0.0
            )
            * 100.0,
            "Rank-1 frequency (%)": np.mean(
                ranks_zero[:, candidate_index] == 1
            )
            * 100.0,
        }
    )

summary_df = pd.DataFrame(summary)
summary_df.to_csv(SUMMARY_PATH, index=False)
 
#%% Supplementary figures - Fig. 8 (bootstrap stability (Figure))

RESULT_PATH = "./bootstrap_output/bootstrap_ensemble_results.npz"

result = np.load(RESULT_PATH)

labels = result["candidate_labels"].astype(str).tolist()
thresholds = result["thresholds"].astype(float)
nominal_delta = result["nominal_delta_y"].astype(float)

zero_index = np.flatnonzero(
    np.isclose(thresholds, 0.0)
)[0]

fractions = result[
    "fraction_by_threshold"
][:, :, zero_index]

ranks = result[
    "ranks_by_threshold"
][:, :, zero_index]

bootstrap_count = fractions.shape[0]
re_index = labels.index("1Re-4Ni")

re_fraction = fractions[:, re_index]

nominal_fraction = (
    np.mean(nominal_delta[re_index] >= 0.0)
    * 100.0
)

median_fraction = np.median(re_fraction)

low_fraction, high_fraction = np.percentile(
    re_fraction,
    [2.5, 97.5],
)

zero_count = np.sum(
    re_fraction == 0.0
)

positive_counts = np.sum(
    fractions > 0.0,
    axis=0,
)

positive_frequency = (
    positive_counts
    / bootstrap_count
    * 100.0
)

rank1_counts = np.sum(
    ranks == 1,
    axis=0,
)

rank1_frequency = (
    rank1_counts
    / bootstrap_count
    * 100.0
)


# =============================================================================
# Figure settings
# =============================================================================
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": 6.3,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.3,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(axis):
    axis.grid(False)

    axis.tick_params(
        direction="out",
        width=0.6,
        length=2.5,
        pad=2,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.65)


def panel_label(axis, label):
    axis.text(
        0.0,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )


mm_to_inch = 1.0 / 25.4

fig = plt.figure(
    figsize=(
        180 * mm_to_inch,
        64 * mm_to_inch,
    ),
    dpi=300,
)

grid = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.28, 1.0, 1.0],
    left=0.065,
    right=0.985,
    bottom=0.185,
    top=0.90,
    wspace=0.38,
)

ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[0, 2])

color_re = "#E69F00"
color_re_dark = "#9E5A00"
color_other = "#8EB3CC"
edge_color = "#3F3F3F"


# =============================================================================
# Panel a
# =============================================================================
bins = np.arange(-2.5, 102.6, 5.0)

counts, _, _ = ax_a.hist(
    re_fraction,
    bins=bins,
    color=color_re,
    edgecolor=edge_color,
    linewidth=0.4,
)

ymax = max(
    float(np.max(counts)),
    1.0,
)

ax_a.vlines(
    nominal_fraction,
    0,
    ymax * 1.06,
    color="#222222",
    linestyle="--",
    linewidth=1.0,
)

ax_a.vlines(
    median_fraction,
    0,
    ymax * 1.06,
    color=color_re_dark,
    linewidth=1.1,
)

bracket_y = ymax * 0.70

ax_a.hlines(
    bracket_y,
    low_fraction,
    high_fraction,
    color=edge_color,
    linewidth=0.8,
)

ax_a.vlines(
    [low_fraction, high_fraction],
    bracket_y - ymax * 0.035,
    bracket_y + ymax * 0.035,
    color=edge_color,
    linewidth=0.8,
)

ax_a.text(
    (low_fraction + high_fraction) / 2.0,
    bracket_y + ymax * 0.06,
    f"95% interval: {low_fraction:.0f}-{high_fraction:.0f}%",
    ha="center",
    va="bottom",
    fontsize=5.5,
)

ax_a.text(
    4.0,
    ymax * 0.50,
    f"Zero region in {zero_count}/{bootstrap_count} refits",
    ha="left",
    va="center",
    fontsize=5.5,
)

ax_a.set_xlim(-2.5, 102.5)
ax_a.set_ylim(0, ymax * 1.34)

ax_a.set_xlabel(
    "Re-Ni positive-region fraction (%)"
)
ax_a.set_ylabel(
    "Number of bootstrap refits"
)

ax_a.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=f"Nominal = {nominal_fraction:.1f}%",
        ),
        Line2D(
            [0],
            [0],
            color=color_re_dark,
            linewidth=1.1,
            label=f"Bootstrap median = {median_fraction:.1f}%",
        ),
    ],
    loc="upper right",
    frameon=False,
    handlelength=2.0,
)

panel_label(ax_a, "a")
style_axis(ax_a)


# =============================================================================
# Panel b
# =============================================================================
x = np.arange(len(labels))

colors = [
    color_re if label == "1Re-4Ni" else color_other
    for label in labels
]

ax_b.bar(
    x,
    positive_frequency,
    width=0.68,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.4,
)

for index, (count, value) in enumerate(
    zip(
        positive_counts,
        positive_frequency,
    )
):
    ax_b.text(
        index,
        value + 1.4,
        f"{value:.1f}%\n({count}/{bootstrap_count})",
        ha="center",
        va="bottom",
        fontsize=5.2,
        fontweight=(
            "bold"
            if labels[index] == "1Re-4Ni"
            else "normal"
        ),
    )

ax_b.set_ylim(0, 61)
ax_b.set_yticks(
    [0, 10, 20, 30, 40, 50, 60]
)

ax_b.set_ylabel(
    "Refits with a non-zero\npositive region (%)"
)

ax_b.set_xticks(x)
ax_b.set_xticklabels(
    labels,
    rotation=38,
    ha="right",
    rotation_mode="anchor",
)

panel_label(ax_b, "b")
style_axis(ax_b)


# =============================================================================
# Panel c
# =============================================================================
ax_c.bar(
    x,
    rank1_frequency,
    width=0.68,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.4,
)

for index, value in enumerate(rank1_frequency):
    ax_c.text(
        index,
        value + 2.0,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=5.3,
        fontweight=(
            "bold"
            if labels[index] == "1Re-4Ni"
            else "normal"
        ),
    )

ax_c.text(
    0.03,
    0.95,
    "Ranked by region fraction;\nmean ΔY resolves ties",
    transform=ax_c.transAxes,
    ha="left",
    va="top",
    fontsize=5.2,
    color="#444444",
)

ax_c.set_ylim(0, 108)
ax_c.set_yticks(
    [0, 20, 40, 60, 80, 100]
)

ax_c.set_ylabel(
    "Rank-1 frequency (%)"
)

ax_c.set_xticks(x)
ax_c.set_xticklabels(
    labels,
    rotation=38,
    ha="right",
    rotation_mode="anchor",
)

panel_label(ax_c, "c")
style_axis(ax_c)

# plt.savefig('./figure_supp_8.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_8.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 17 (Process-level cost breakdown)
# ── Nature figure settings ────────────────────────────────
plt.rcParams["font.family"]        = "sans-serif"
plt.rcParams["font.sans-serif"]    = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"]     = 0.8
plt.rcParams["pdf.fonttype"]       = 42   
plt.rcParams["ps.fonttype"]        = 42

# Font sizes (pt) 
FS_LABEL = 8     
FS_AXIS  = 7      
FS_TICK  = 6     
FS_DATA  = 6     
FS_LEG   = 6.5   

cases = ["Ni-Re", "Pd"]

OPERATING_HOURS = 8000

raw_cost = pd.DataFrame({
    "Ni-Re": {
        "Furfural":  76_868_480,
        "H2":        13_304_808,
        "IPA":        6_249_975.68,
        "Catalyst":     510_720,
    },
    "Pd": {
        "Furfural":  76_868_480,
        "H2":        13_304_808,
        "IPA":        6_249_975.68,
        "Catalyst":  33_951_168,
    }
})

summary = pd.DataFrame({
    "THFA production (10$^3$ ton/yr)": {
        "Ni-Re": 68_875 / 1e3,
        "Pd":    71_027 / 1e3,
    },
    "Utility cost (M$/yr)": {
        "Ni-Re": 10_396_138 / 1e6,
        "Pd":     9_887_154 / 1e6,
    },
    "CAPEX (M$)": {
        "Ni-Re": 9_475_630 / 1e6,
        "Pd":     8_985_560 / 1e6,
    }
})

colors_raw = {
    "Furfural":  "#4C78A8",
    "H2":        "#F58518",
    "IPA":       "#B279A2",
    "Catalyst":  "#54A24B",
}
legend_labels = {
    "Furfural": "Furfural",
    "H2":       "H$_2$",
    "IPA":      "IPA",
    "Catalyst": "Catalyst",
}
colors_case = {
    "Ni-Re": "#0F7C80",
    "Pd":    "#B8BEC8",
}

fig = plt.figure(figsize=(7.09, 3.9), dpi=300)
gs = GridSpec(
    nrows=3, ncols=2, figure=fig,
    width_ratios=[1.3, 1.0],
    height_ratios=[1, 1, 1],
    wspace=0.30, hspace=0.85
)

ax_raw       = fig.add_subplot(gs[:, 0])
axes_summary = [fig.add_subplot(gs[i, 1]) for i in range(3)]

# ── Panel a ────────────────────────────────────────────────
x         = np.arange(len(cases))
bar_width = 0.55
bottom    = np.zeros(len(cases))
totals    = raw_cost[cases].sum(axis=0).values

for item in raw_cost.index:
    values = raw_cost.loc[item, cases].values.astype(float)
    ax_raw.bar(
        x, values / 1e6,
        bottom=bottom / 1e6,
        width=bar_width,
        color=colors_raw[item],
        edgecolor="white",
        linewidth=0.6,
        label=legend_labels[item]
    )
    percents = values / totals * 100
    for i, (v, p) in enumerate(zip(values, percents)):
        if p >= 3:
            ax_raw.text(
                x[i], (bottom[i] + v / 2) / 1e6,
                f"{p:.1f}%",
                ha="center", va="center",
                fontsize=FS_DATA, color="black"
            )
    bottom += values

for i, total in enumerate(totals / 1e6):
    ax_raw.text(
        x[i], total + 3.0,
        f"{total:.2f} M$/yr",
        ha="center", va="bottom",
        fontsize=FS_DATA
    )

ax_raw.set_ylabel("Raw material cost (M$/yr)", fontsize=FS_AXIS)
ax_raw.set_xticks(x)
ax_raw.set_xticklabels(["1Re-4Ni", "5Pd"])
ax_raw.set_ylim(0, max(totals / 1e6) * 1.18)
ax_raw.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK)
ax_raw.spines["top"].set_visible(False)
ax_raw.spines["right"].set_visible(False)
ax_raw.legend(
    frameon=False, fontsize=FS_LEG, loc="upper left",
    bbox_to_anchor=(0.02, 0.98), handlelength=1.2, handletextpad=0.5
)
ax_raw.set_title("a", loc="left", fontsize=FS_LABEL, fontweight="bold", pad=4)

# ── Panels b-d ─────────────────────────────────────────────
panel_letters = ["b", "c", "d"]

for ax, metric, letter in zip(axes_summary, summary.columns, panel_letters):
    vals  = summary[metric].loc[cases].values
    y_pos = np.arange(len(cases))

    bars = ax.barh(
        y_pos, vals, height=0.5,
        color=[colors_case[c] for c in cases],
        edgecolor="black", linewidth=0.5
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["1Re-4Ni", "5Pd"])
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        fmt = f"{v:.3f}" if metric == "CAPEX (M$)" else f"{v:.2f}"
        ax.text(
            bar.get_width() + max(vals) * 0.03,
            bar.get_y() + bar.get_height() / 2,
            fmt, ha="left", va="center", fontsize=FS_DATA
        )

    ax.set_xlabel(metric, fontsize=FS_AXIS)          # 지표명 → x축 라벨 (단위 포함)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(letter, loc="left", fontsize=FS_LABEL, fontweight="bold", pad=4)

fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=1.2)

# plt.savefig('./figure_supp_17.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_17.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 18 (Tornado sensitivity analysis)
# =========================
# Style — Nature figure spec
# =========================
plt.rcParams['font.family']        = 'sans-serif'     
plt.rcParams['font.sans-serif']    = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth']     = 0.8
plt.rcParams['pdf.fonttype']       = 42
plt.rcParams['ps.fonttype']        = 42

FS_LABEL = 8      
FS_AXIS  = 7      
FS_TICK  = 6     
FS_DATA  = 5.5    
FS_LEG   = 6.5   
FS_BASE  = 6.5   

# =========================
# Sensitivity data
# =========================
data = {
    "Pd": {
        "base": 4019.23,
        "FF price":       {"low": 3581.30, "high": 4457.26},
        "Catalyst price": {"low": 3825.80, "high": 4212.66},
        "H$_2$ price":    {"low": 3943.43, "high": 4095.03},
        "Utility cost":   {"low": 3962.90, "high": 4075.56},
        "IPA price":      {"low": 3983.62, "high": 4054.84},
    },
    "Ni-Re": {
        "base": 3178.83,
        "FF price":       {"low": 2727.22, "high": 3630.45},
        "Catalyst price": {"low": 3175.41, "high": 3181.41},
        "H$_2$ price":    {"low": 3100.41, "high": 3256.41},
        "Utility cost":   {"low": 3117.41, "high": 3239.91},
        "IPA price":      {"low": 3142.11, "high": 3215.41},
    }
}

PARAM_ORDER = [
    "FF price",
    "Catalyst price",
    "H$_2$ price",
    "Utility cost",
    "IPA price",
]
# =========================
# Tornado plot
# =========================
def plot_tornado(ax, case_name, case_data, letter):
    base = case_data["base"]

    rows = []
    for param, values in case_data.items():
        if param == "base":
            continue
        low_delta  = values["low"]  - base
        high_delta = values["high"] - base
        rows.append({
            "Parameter": param,
            "Low MSP":   values["low"],
            "High MSP":  values["high"],
            "Low delta":  low_delta,
            "High delta": high_delta,
            "Range": abs(high_delta - low_delta)
        })

    df = (pd.DataFrame(rows)
            .set_index("Parameter")
            .loc[PARAM_ORDER[::-1]]
            .reset_index())
    y = np.arange(len(df))

    ax.barh(y, df["Low delta"],  color="#4C78A8", alpha=0.90, height=0.62, label="-20%")
    ax.barh(y, df["High delta"], color="#F58518", alpha=0.90, height=0.62, label="+20%")
    ax.axvline(0, ymax=0.86, color="black", linewidth=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Parameter"], fontsize=FS_TICK)
    ax.tick_params(axis="x", direction="in", length=3, width=0.8, labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0)    

    ax.set_xlabel(r"MSP deviation from base case (\$/ton THFA)",
                  fontsize=FS_AXIS, labelpad=4)
    ax.set_title(letter, loc="left", fontsize=FS_LABEL, fontweight="bold", pad=6)

    ax.text(0, len(df) - 1 + 0.7,
            rf"{case_name}   Base MSP = {base:,.0f} \$/ton THFA",
            ha="center", va="bottom", fontsize=FS_BASE)
    x_span = max(abs(df["Low delta"]).max(), abs(df["High delta"]).max())
    offset = x_span * 0.04
    for i, row in df.iterrows():
        ax.text(row["Low delta"]  - offset, i, f"{row['Low MSP']:,.0f}",
                va="center", ha="right", fontsize=FS_DATA)
        ax.text(row["High delta"] + offset, i, f"{row['High MSP']:,.0f}",
                va="center", ha="left",  fontsize=FS_DATA)

    ax.set_xlim(-x_span * 1.34, x_span * 1.34)
    ax.set_ylim(-0.7, len(df) - 1 + 1.1)          
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.3), dpi=300)

df_pd   = plot_tornado(axes[0], "5Pd",     data["Pd"],    "a")
df_nire = plot_tornado(axes[1], "1Re-4Ni", data["Ni-Re"], "b")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           fontsize=FS_LEG, bbox_to_anchor=(0.5, 1.02),
           handlelength=1.2, handletextpad=0.5, columnspacing=1.5)

fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.5)

# plt.savefig('./figure_supp_18.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_18.pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Supplementary figures - Fig. 19 (MSP sensitivity analysis)

plt.rcParams['font.family']        = 'sans-serif'
plt.rcParams['font.sans-serif']    = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth']     = 0.8
plt.rcParams['pdf.fonttype']       = 42
plt.rcParams['ps.fonttype']        = 42

FS_AXIS = 7
FS_TICK = 6
FS_LEG  = 6.5
FS_BASE = 6  

# =========================
# Catalyst lifetime sensitivity data
# =========================
lifetime = np.array([0.5, 1, 2, 3])
msp_nire = [3193.84, 3178.83, 3171.33, 3168.83]
msp_pd   = [4986.36, 4019.23, 3535.67, 3374.16]
# =========================
# Line plot — single column 89 mm ≈ 3.5 in
# =========================
fig, ax = plt.subplots(figsize=(3.35, 2.9), dpi=300)


ax.axvline(1.0, color="0.55", linewidth=0.8, linestyle="--", zorder=0)

ax.plot(lifetime, msp_pd, marker="o", markersize=4.5, linewidth=1.0,
        color="#EE854A", label="5Pd/Al$_2$O$_3$", zorder=3)
ax.plot(lifetime, msp_nire, marker="*", markersize=7, linewidth=1.0,
        color="#2166AC", label="1Re-4Ni/Al$_2$O$_3$", zorder=3)

ax.set_xlabel("Catalyst lifetime (years)", fontsize=FS_AXIS, labelpad=4)
ax.set_ylabel(r"MSP (\$/ton THFA)", fontsize=FS_AXIS, labelpad=4)

ax.set_xticks(lifetime)
ax.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax.text(1.0, ax.get_ylim()[1], " Baseline", ha="left", va="top",
        fontsize=FS_BASE, color="0.35")
ax.legend(frameon=False, fontsize=FS_LEG, loc="upper right",
          handlelength=1.5, handletextpad=0.5)

fig.tight_layout()

# plt.savefig('./figure_supp_19.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_19.pdf', dpi=600, bbox_inches='tight')
plt.show()