import pandas as pd
import numpy as np
import base64
import os
import torch 
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import glob
from sklearn.preprocessing import MinMaxScaler
from deep_one_class.src.optim.ae_trainer import bidirectional_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from deep_one_class.src import deepSVDD
from deep_one_class.visualizations import plots
from deep_one_class.src.deepSVDD import *
from deep_one_class.src.utils.config import Config
#from mordred import Calculator, descriptors  #https://github.com/mordred-descriptor/mordred
calc = Calculator(descriptors, ignore_3D=True)
import joblib
import pickle
import subprocess

cfg = Config({'nu': 0.05, 
              'objective':  'one-class'} ) 

def get_representation(smiles1, smiles2):
    """ Given the smiles of a validation dataset convert it to fingerprint
     representation """   
    df = pd.concat([pd.DataFrame(smiles1),
     pd.DataFrame(smiles2)], axis=1)
    return df

def smiles2txt(dataset):  
    ''' reading the smiles from the csv file and saves them in txt file in order to get the 
    graph embendings from each smile'''

    with open(os.path.join("gnn",'smiles1.txt'), 'w') as f:
        for item in dataset['smiles1'].values:
            f.write("%s\n" % item)

    with open(os.path.join("gnn", 'smiles2.txt'), 'w') as f:
        for item in dataset['smiles2'].values:
            f.write("%s\n" % item)

def ae_score(deep_SVDD, X):
    device =  'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        net = deep_SVDD.ae_net.to(device)
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        scores = bidirectional_score(X, y)
    return scores

    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class PairsEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 50
        self.seq = nn.Sequential(SAB(dim_in=300, dim_out=150, num_heads=5),
              SAB(dim_in=150, dim_out=50, num_heads=5),
            PMA(dim=50, num_heads=2, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp, 300, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

    def get_attention_weights(self):
        return [layer.get_attention_weights() for layer in self.seq]

class PairsAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=150), nn.LeakyReLU(),
                                     nn.Linear(in_features=150, out_features=300), nn.LeakyReLU(),
        nn.Linear(in_features=300, out_features=600))
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_attention_weights(self):
        return self.encoder.get_attention_weights()

def build_autoencoder(net_name):
    return PairsAutoEncoder()

def build_network(net_name):  
    return PairsEncoder()

def score(deep_SVDD, X):
    with torch.no_grad():
        device = 'cpu'  
        net = deep_SVDD.net.to(device)
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        c, R = torch.FloatTensor([deep_SVDD.c]).to(device), torch.FloatTensor([deep_SVDD.R]).to(device)
        dist = torch.sum((y - c)**2, dim=1)
        if deep_SVDD.objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
    return scores

def gnn_score_dropout(smiles1, smiles2):
   
    validation_set= get_representation(smiles1, smiles2)    
    #dataset=dataset[~dataset.smiles1.isin(wrong_smiles)]
    #dataset=dataset[~dataset.smiles2.isin(wrong_smiles)]

    smiles2txt(validation_set)
    subprocess.call("python gnn/main.py -fi gnn/smiles1.txt -m gin_supervised_masking -o gnn/results1", shell=True)
    subprocess.call("python gnn/main.py -fi gnn/smiles2.txt -m gin_supervised_masking -o gnn/results2", shell=True)
    valid1 = np.load('gnn/results1/mol_emb.npy')
    valid2 = np.load('gnn/results2/mol_emb.npy')
    #print(valid1.shape)
    validation_set = pd.concat([pd.DataFrame(valid1), pd.DataFrame(valid2)],axis=1)
    torch.manual_seed(0)
        
    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder = build_autoencoder
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    net_name='fingerprint_checkpoint.pth'
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    #model_path='https://github.com/katerinavr/streamlit/releases/download/weights/model_150_1e-3_64_1e-05_fingerprint.pth'
    #deep_SVDD.load_model(model_path, True)
    deep_SVDD.load_model('deep_one_class/model_300_1e-3_32_1e-05_gnn.pth', True) 
    #print(deep_SVDD)
    X=validation_set.iloc[:,:].values
    with torch.no_grad():
        torch.manual_seed(0)
        result = []
        for i in range(10):
            net = deep_SVDD.ae_net.to('cpu')
            X = torch.FloatTensor(X).to('cpu')
            #print(X)
            y = net(X)#.to(device)
            #print(y)
            #print(y.dim(), y.shape, X.shape)
            scores = -1*bidirectional_score(X, y)#torch.sum((y - X) ** 2, dim=tuple(range(1, y.dim())))
            scores = scores.clip(-50,0)
            scaler = MinMaxScaler()
            #lab = -1*ae_score(deep_SVDD, df_paws.iloc[:,:].values).cpu().detach().numpy() #
            #lab = lab.clip(-50,0)
            #lab1 = X_scaler.fit_transform(lab.reshape(-1,1))
            #scores=X_scaler.transform(scores.reshape(-1, 1)).ravel()
            scores=scaler.fit_transform(scores.reshape(-1, 1)).ravel()
            result.append(scores)     
    return np.mean(result, axis=0), np.std(result, axis=0) 
