import streamlit as st
import pandas as pd
import numpy as np
import base64
from rdkit import Chem
from mordred import Calculator, descriptors

def mordred_descriptors():
    sentence = st.text_area('Input your SMILES here:') 
    calc = Calculator(descriptors, ignore_3D=True)
    if len(sentence.split()) > 0:
        mol = [Chem.MolFromSmiles(x) for x in sentence.split()]
        descriptors_mol1 =[]
        for mol in sentence.split():
    	    try:
                descriptors_mol1.append(calc(Chem.MolFromSmiles(mol)))
    	    except TypeError:
                descriptors_mol1.append('none')
        dataset1 = pd.DataFrame(descriptors_mol1)
        df1 = pd.DataFrame(dataset1.values, columns=calc.descriptors)#.to_csv('dataset1.csv', index=False)
        df = pd.concat([pd.DataFrame(sentence.split(), columns=['smiles']), df1], axis=1)
        st.write(df)
        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(df, 'descriptors.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'