import pandas as pd
import numpy as np
import base64
import os
import torch 
from rdkit import Chem
from rdkit.Chem import AllChem
import glob
from sklearn.preprocessing import MinMaxScaler
from deep_one_class.src.optim.ae_trainer import bidirectional_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from deep_one_class.src import deepSVDD
#from deep_one_class.visualizations import plots
from deep_one_class.src.deepSVDD import *
from deep_one_class.src.utils.config import Config
from deep_one_class.utils import *
import argparse
import wget
model_link='https://github.com/katerinavr/streamlit/releases/download/weights/model_150_1e-3_64_1e-05_fingerprint.pth'


cfg = Config({'nu': 0.05, 
              'objective':  'one-class'} ) 

def get_representation(smiles1, smiles2):
    """ Given the smiles of a validation dataset convert it to fingerprint
     representation """   
    df = pd.concat([fingerprint_from_df(smiles1, 'bits_1'),
     fingerprint_from_df(smiles2, 'bits_2')], axis=1)
    return df

def smile_to_fingerprint(smile):
  mol = Chem.MolFromSmiles(smile)
  return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096, useChirality=True)

def fingerprint(smiles):
  bits = []
  for smile in smiles:
    bits.append(np.asarray(smile_to_fingerprint(smile)))
  return bits

def fingerprint_from_df(smiles, prefix):
  df = pd.DataFrame(fingerprint(smiles))
  columns = [f'{prefix}_{i}' for i in df.columns]
  df.columns = columns
  return df

def ae_score(deep_SVDD, X):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def __init__(self,proba=0.1):
        super().__init__()
        self.rep_dim = 50
        self.seq = nn.Sequential(SAB(dim_in=4096, dim_out=500, num_heads=10),
              #SAB(dim_in=1500, dim_out=500, num_heads=2),
              nn.Dropout(p=proba),
              SAB(dim_in=500, dim_out=50, num_heads=10),
              nn.Dropout(p=proba),
            PMA(dim=50, num_heads=10, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp, 4096, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

class PairsAutoEncoder(BaseNet):

    def __init__(self,proba=0.1):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=500), nn.LeakyReLU(),
                                     nn.Dropout(p=proba),
                                     nn.Linear(in_features=500, out_features=4096), nn.LeakyReLU(),
                                     nn.Dropout(p=proba),
                                     #nn.Linear(in_features=1000, out_features=4096), nn.LeakyReLU(),
                                     #nn.Dropout(p=proba),
        nn.Linear(in_features=4096, out_features=8192))#, nn.Sigmoid())# ,nn.LeakyReLU()
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

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

def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Dropout'):
      each_module.train()

def ae_score_dropout(smiles1, smiles2):
  
  validation_set = get_representation(smiles1, smiles2)
  torch.manual_seed(0)
    
  deepSVDD.build_network = build_network
  deepSVDD.build_autoencoder = build_autoencoder
  deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
  net_name='fingerprint_checkpoint.pth'
  deep_SVDD.set_network(net_name)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'    
  model_path = 'deep_one_class/model_fingerprint.pth'
  download_file(model_path, model_link)
  deep_SVDD.load_model(model_path, True) 
  
  #ecfp4_model = wget.download(model_path)
  deep_SVDD.load_model(model_path, True)
  #deep_SVDD.load_model('deep_one_class/model_150_1e-3_64_1e-05_fingerprint.pth', True) 
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