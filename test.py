import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors


def naki():
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
	#df.to_csv(index=False)
        return st.dataframe(df)
    	#st.markdown(get_table_download_link(df), unsafe_allow_html=True)

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """


    # Import the unlabeled dataset
    unlabeled = pd.read_csv('data/zinc15_dataset.csv')
    val = unlabeled['Identifier'].values
    length = len(val)
    pairs = [[val[i],val[j]] for i in range(length) for j in range(length) if i!=j ]
    # Remove the duplicate structures
    no_dups = []
    for pair in pairs:
        if not any(all(i in p for i in pair) for p in no_dups):
            no_dups.append(pair)
    pairs = pd.DataFrame(no_dups)
    keys = unlabeled['Identifier'].values
    values = unlabeled.iloc[:, 2:].values
    d = {key:value for key, value in zip(keys, values)}
    mol1_data= list()
    for mol1 in pairs[0]:       
        mol1_data.append(d[mol1])    
    mol1_data = pd.DataFrame(mol1_data, columns = unlabeled.iloc[:, 2:].columns.values)   
    mol2_data= list()
    for mol2 in pairs[1]:   
        mol2_data.append(d[mol2])
    mol2_data = pd.DataFrame(mol2_data, columns= unlabeled.iloc[:, 2:].columns.values) 
    final_1 = pd.concat([pairs[0],mol1_data],axis=1)
    final_1 = final_1.fillna(0)
    final_2 = pd.concat([pairs[1],mol2_data],axis=1)
    final_2 = final_2.fillna(0)
    unlab=pd.concat([pairs[0], pairs[1]], axis=1)
    final_1 = final_1.replace({'#NUM!': 0})
    final_2 = final_2.replace({'#NUM!': 0})
    final_11 = final_1.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    final_22 = final_2.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    # Standarize the unlabeled data based on the labelled
    final_1_scaled = pd.DataFrame(X_scaler.transform(final_11.iloc[:,1:]))
    final_2_scaled = pd.DataFrame(X_scaler.transform(final_22.iloc[:,1:]))
    uf=pd.concat([final_1_scaled, final_2_scaled], axis =1)