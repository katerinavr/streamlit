# Plottly graph
import base64
import io
import json
from base64 import b64encode
from io import BytesIO
import numpy as np
from PIL import Image
import dash
from dash import html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import streamlit as st
from dash.dependencies import Input, Output
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PandasTools import ChangeMoleculeRendering
import plotly.graph_objects as go


def plottly_raking(df):
    # set the style of the webpage
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__,
            external_stylesheets=external_stylesheets)
    server = app.server

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    df.sort_values(by='score', ascending=False, inplace=True)
    df['rank'] = np.arange(len(df.score))
 
    df['color'] = 'blue'
    df.color[df.uncertainty > 0.1] = 'red'

    st.subheader("Click on the points to visualize the pairs")
    st.markdown("Molecular pairs with high-uncertainty are shown in red")
    fig = px.scatter(df , x='rank', y = 'score', error_y="uncertainty", color="color", range_x=[0, 1]
    color_discrete_sequence=["blue", "red"], hover_data=["smiles1","smiles2","score"])
    fig.layout.update(showlegend=False)

    plot_name_holder = st.empty()
    clicked_point = plotly_events(fig, click_event=True, hover_event=False)

    if len(clicked_point) > 0:
        smiles_1 = df.iloc[clicked_point[0]['x']]['smiles1']
        m1 = Chem.MolFromSmiles(smiles_1)
        im1=Draw.MolToImage(m1)

        smiles_2 = df.iloc[clicked_point[0]['x']]['smiles2']
        m2 = Chem.MolFromSmiles(smiles_2)
        im2=Draw.MolToImage(m2)
        st.image([im1, im2], caption=[df.iloc[clicked_point[0]['x']]['smiles1'],df.iloc[clicked_point[0]['x']]['smiles2']])