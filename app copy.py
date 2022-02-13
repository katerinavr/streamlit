import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import base64
import os, urllib
#from descriptors import *
#from one_class import *
#import torch 
#from pairs_explorer import *

def main():
    # Render the readme as markdown using st.markdown.
    #readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    #intro_figure = "Picture1.png"
    #st.sidebar.title("Co-Crystal-Net")
    app_mode = st.sidebar.selectbox("Choose your preference",
        ["Show instructions", "Calculate descriptors", "Deep One Class Classification", 
        "Interprete the model", "Pairs Explorer", "Design novel pairs"])        
    #if app_mode == "Show instructions":
        #st.sidebar.success('To continue select "Run the app".')
    #    st.image(intro_figure, use_column_width=True) 
    #elif app_mode == "Calculate descriptors":
    #    readme_text.empty()
    #    mordred_descriptors()
    #elif app_mode == "Deep One Class Classification":
    #    plot_scores()
        #tza()
        #load_data()
        #train_and_score()
        #st.sidebar.success('To continue select "Run the app".')
    #elif app_mode == "Interprete the model":
    #    readme_text.empty()
    #elif app_mode == "Pairs Explorer":
    #    readme_text.empty()
        #df = pd.read_csv('data/dataset1.csv')
        #st.dataframe(df)
     #   predicted_pairs()
    #elif app_mode == "Design novel pairs":
    #    readme_text.empty()
    #    st.write(readme_text.read())

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/katerinavr/streamlit/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

if __name__ == "__main__":
    main()
