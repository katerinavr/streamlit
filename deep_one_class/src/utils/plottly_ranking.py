# Plottly graph
import base64
import io
import json
from base64 import b64encode
from io import BytesIO
import numpy as np
from PIL import Image
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import streamlit as st
from dash.dependencies import Input, Output
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw



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

    fig = px.scatter(df , x='rank', y = 'score', custom_data=["smiles1","smiles2","score"])
    fig.update_traces(marker=dict(size=10,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'),
                  marker_size=20)
    fig.update_layout(clickmode='event+select')


    grid_style = {"border-radius": "4px", "margin": "2px","text-align": "center"}
    header_style = {"background-color": "lightgrey"}
    header_style.update(grid_style)
    cell_style = {"padding": "130px 0"}
    cell_style.update(grid_style)

    # Div is a block in html
    app.layout = html.Div([
        dcc.Graph(
            id='main_plot',
            figure=fig),

        html.Div(id='table', children=[
            # Header
            html.Div(className='row', children=[
                html.Div([html.P("Molecule 1")], className="five columns", style=header_style),
                html.Div([html.P("Molecule 2")], className="five columns", style=header_style),
                html.Div([html.P("Score")], className="one columns", style=header_style),
                #html.Div([html.P("Uncertainty")], className="two columns", style=header_style),
            ]),
            # id is the name of the element to be displayed in html
            html.Div(className='row', children=[
                html.Div([
                    html.Img(id='out-smile-1', src=app.get_asset_url('0_4.png')), #random
                ], className='five columns', style=grid_style),

                html.Div([
                    html.Img(id='out-smile-2', src=app.get_asset_url('0_4.png')),
                ], className='five columns', style=grid_style),
                html.Div([html.P("0.5", id="score")], className="one columns", style=cell_style),
                #html.Div([html.P("0.5", id="uncertainty")], className="two columns", style=cell_style),
            ])
        ])
    ], className="container")

    # wheverer a click occurs call the display_click_data()
    # clickData is a dictionary automatically created by plottly
    # points is a list of points that were clicked

    @app.callback(
        Output('out-smile-1', 'src'),
        Output('out-smile-2', 'src'),
        Output('score', 'children'),
        #Output('uncertainty', 'children'),
        Input('main_plot', 'clickData'))


    def generate_image(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.Draw.MolToImage(mol, size=(300,300))

    def display_click_data(clickData):
        if clickData:
            smiles1, smiles2, score, uncertainty = clickData['points'][0]['customdata']
            img1 = generate_image(smiles1)
            img2 = generate_image(smiles2)
            return format_img(img1),format_img(img2), f"{score:.2f}"#, f"{uncertainty:.2f}"
        return [None, None, "", ""]

    # Takes a pil image and returns the encoded version for display in html
    def format_img(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return "data:image/png;base64," + img_str.decode('utf-8')

    st.plotly_chart(fig, use_container_width=True)


