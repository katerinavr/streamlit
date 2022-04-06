﻿import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import base64
import os, urllib
from rdkit import Chem
from deep_one_class.ecfp4 import *
from deep_one_class.gnn import *
from deep_one_class.src.utils.ranking_plot import *
from deep_one_class.src.utils.plottly_ranking import *
from deep_one_class.src.deepSVDD import *

col1,col2= st.columns([1,1])

with col1:
    '''Title the app'''
    st.title('Molecular Set Transformer')
    st.markdown('#### for co-crystal screening')

with col2:
    intro_figure = "image.png"
    st.image(intro_figure, use_column_width=True) 

st.markdown("""
    * To see the map of the existing co-crystals click here : https://csd-cocrystals.herokuapp.com
    * Use the menu at left to input the molecular pairs and select the model
    * Press the Predict button
    * Your plots will appear below
    * Default dataset: artemisinin co-crystals ranking
    """)
st.sidebar.markdown("## Define molecular pairs")
smiles1 = st.sidebar.text_area('Input a list of SMILES as the first coformer (API):') 
smiles2 = st.sidebar.text_area('Input a list of SMILES as the second coformer (excipient):') 
st.sidebar.markdown("## Pretrained models")
model= st.sidebar.selectbox('Select the machine learning model',
                                    ['ECFP4', 'GNN'])


def rank_pairs(smiles1, smiles2, model):
    ''' Select between the gnn and ecfp4 models
    and get a score and the uncertainty for any given molecular pair ''' 
    # TODO: check if a valid smiles is given
    if model == 'GNN':
        scores, uncertainty= gnn_score_dropout(smiles1, smiles2) 
    
    else: 
        scores, uncertainty= ae_score_dropout(smiles1, smiles2) 

    df = pd.concat([pd.DataFrame(smiles1, columns=['smiles1']), pd.DataFrame(smiles2, columns=['smiles2']),
    pd.DataFrame(scores, columns=['score']), pd.DataFrame(uncertainty, columns=['uncertainty'])], axis=1)
    return df
   
if st.sidebar.button('Predict!'):
    df = rank_pairs(smiles1.split(), smiles2.split(), model)
    df.to_csv('data/ranking.csv', index=None)
    
df = pd.read_csv('data/ranking.csv')
st.write(df)
with open('data/ranking.csv') as f:
        st.download_button('Download Table as CSV', f)
plottly_raking(df)


