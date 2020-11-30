""" 
The user can select one of the molecules out of a provided list and visualize the top pairs in a 
dataframe
Then all the posible pairs are projected to two dimensional plots (x any y dimensions are user selected)
and can be further optimized with Pareto optimization
"""
import streamlit as st
from sklearn import datasets, metrics
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import pandas as pd
import os
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import rdDepictor
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PandasTools import ChangeMoleculeRendering

def moltoimage(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol, size=(600,600))

def predicted_pairs():
    zinc_dataset = pd.read_csv('data/zinc15_dataset.csv')
    total_scores = pd.read_csv('data/final_scores.csv')
    
    na = zinc_dataset.iloc[:, :2].head()
    PandasTools.AddMoleculeColumnToFrame(na,'smiles','Molecule',includeFingerprints=True)
    st.write(na, unsafe_allow_html=True)
    #st.write('Here you can visualize the predicted pairs and optimize your selection based on some electronic properties')
    
    #endpoints = zinc_dataset.smiles.values
    #selected_endpoint = st.selectbox('Select a molecule of interest to find the top scored co-formers',endpoints)
    #print(selected_endpoint)
    #st.write(selected_endpoint)
    #coformer=[]
    #score=[]
    #molecules = total_scores.mol2[total_scores.mol1 == selected_endpoint].values + total_scores.mol1[total_scores.mol2 == selected_endpoint].values
    #for i in molecules:
    #    try:
    #        score.append(total_scores.score[total_scores.mol1 == i].values)
    #    except ValueError:
    #        score.append(total_scores.score[total_scores.mol1 == i].values)
    #score.append([total_scores.score[total_scores.mol2 == i] for i in molecules])
    #for i in molecules: 
      #  if i in total_scores.mol1.values:
     #       score.append(i) #total_scores.score[total_scores.mol1 == i])

    #df= pd.concat([pd.DataFrame(molecules , columns=['Smiles']), 
    #pd.DataFrame(score , columns=['score'])],axis=1)
    #df= pd.DataFrame(molecules , columns=['Smiles'])
    #PandasTools.AddMoleculeColumnToFrame(df,'Smiles','Molecule',includeFingerprints=True)
    #print(df)
    #mols = [Chem.MolFromSmiles(i)  for i in df['Smiles'].values] 
    #for i in molecules:
    #    st.image(moltoimage(i))
    #df['Smiles'].values[0]))
    #st.write(df['Smiles'].values[0])
    #st.image(Draw.MolsToGridImage(mols))
    #st.write(score, unsafe_allow_html=True)

    if st.checkbox('Optimize Selection'):
        st.subheader('Pareto Optimization based on selected property')
        pareto(molecules)


def pareto(molecules):
    zinc_dataset = pd.read_csv('data/zinc15_dataset.csv')
    band_energies = pd.read_csv('data/single_known_bands.csv')
    #st.write(zinc_dataset.head())
    total_scores = pd.read_csv('data/final_scores.csv')
    endpoints = ['LUMO', 'HOMO', 'HOMO-LUMO gap', 'LUMO degeneracy']
    selected_endpoint = st.selectbox('Select feature for optimization',endpoints)
    band_energies[band_energies.Id.isin(molecules)]
    total_scores.score[total_scores.mol1.isin(molecules)]# or total_scores.mol2.isin(molecules)]
    
#    reg_metrics = ['RMSE', 'R2', 'MAE', 'MSE']
#    binary_metrics = ['Accuracy', 'Balance Accuracy', 'F1_score', 'MCC', 'AUROC']
#    multiclass_metrics = ['Accuracy', 'Balance Accuracy', 'F1_score', 'MCC']