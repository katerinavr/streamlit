import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.lines import Line2D
import streamlit as st
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams["font.weight"] = "light"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 16})

def ranking_plot(scores, uncertainty):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set(adjustable='box')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis = 'both', which='both', width=2)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis = 'both', which='major', length=6)
    ax.tick_params(axis = 'both', which='both' , bottom=True, top=True, left=True, right=True, direction='in')
    ax.axis('on')
    ax.grid(False)
    ax.set_facecolor('xkcd:white')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('score', fontsize=14)
    plt.xlabel('rank', fontsize=14)
    #labels = true_label
    pub_validation=pd.DataFrame(scores, columns=['score'])
    pub_val_sort = pub_validation.sort_values(by='score', ascending=False) 
    x=np.arange(len(scores))
    ax.scatter(x ,pub_val_sort.score)
    plt.errorbar(x ,pub_val_sort.score, yerr=uncertainty, fmt="o")
    #plt.title('Validation dataset')
    st.pyplot(fig)