# Import the basic libraries
import streamlit as st
from sklearn import datasets, metrics
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from scipy.spatial.distance import squareform
from matplotlib import cm
import itertools
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import sys
sys.path.extend(['deep_one_class', 'deep_one_class/src/set_transformer', 'deep_one_class.src.base' ])
from deep_one_class.src.set_transformer.modules import SAB, PMA, ISAB
import tqdm
from deep_one_class.src.base.torchvision_dataset import TorchvisionDataset
import logging
import random
from deep_one_class.src.utils.config import Config
from deep_one_class.src import deepSVDD
from deep_one_class.src.deepSVDD import *
from deep_one_class.src.base.base_net import BaseNet
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import figure

def load_data():
    dataset1 = pd.read_csv('data/dataset1.csv')
    dataset2 = pd.read_csv('data/dataset2.csv')
    df1=dataset1.iloc[:,2:]
    df1 = df1.fillna(0)
    df2=dataset2.iloc[:,2:]
    df2 = df2.fillna(0)
    df1 = df1.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    df2 = df2.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)  
    X_scaler = MinMaxScaler()
    df_concat = pd.concat([df1, df2])
    df_concat = df_concat.drop_duplicates(keep='first')
    numerical_cols = df_concat.columns[:]
    df_scaled = pd.DataFrame(X_scaler.fit(df_concat), columns=numerical_cols, index=df_concat.index)
    numerical_cols = df2.columns[:]
    df1_scaled =  pd.DataFrame(X_scaler.transform(df1[numerical_cols]), columns=numerical_cols, index=df1.index)
    df2_scaled = pd.DataFrame(X_scaler.transform(df2[numerical_cols]), columns=numerical_cols, index=df2.index)
    # Final bidirectional concatenated dataset, after feature selection and scaling 
    df = concat_bidirectional(df1_scaled,df2_scaled)
    labelled = pd.concat([df1_scaled, df2_scaled], axis=1)
    return labelled
    
        
def concat_bidirectional(dataset11, dataset22):
    dataset1 = pd.read_csv('data/dataset1.csv')
    dataset2 = pd.read_csv('data/dataset2.csv')
    return pd.concat([pd.concat([dataset1['id'], dataset11, dataset22], axis=1), pd.concat([dataset1['id'].apply(lambda x: f"{x}_"),dataset22, dataset11], axis=1) ])

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

def train_and_score():
    cfg = Config({'normal_class': 1, 
              'n_jobs_dataloader': 0, 
              'ae_weight_decay': 0.0005, 
              'ae_batch_size': 200, 
              'ae_lr_milestone': (50,), 
              'ae_n_epochs': 5, 
              'ae_lr': 0.0001,
              'ae_optimizer_name': 'adam', 
              'pretrain': True, 
              'weight_decay': 5e-07,
              'batch_size': 200, 
              'lr_milestone': (50,),
              'n_epochs': 5, 
              'lr': 0.0001, 
              'optimizer_name': 'adam', 
              'seed': 0, 
              'device': 'cuda', 
              'nu': 0.05, 
              'objective':  'one-class', 
              'load_model': None, 
              'load_config': None,'dataset_name': 'cocry', 'net_name': 'CocryNet'} ) 
    #set_seed()
    lab_list=[]
    unlab_list=[]
    labelled = load_data()
    net_name = cfg.settings['net_name']
    #st.write(labelled.head())
    dataset = Pairs_Dataset('', data= labelled )
    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder =  build_autoencoder
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(net_name)
    device = 'cpu'                      
    deep_SVDD.pretrain(dataset, optimizer_name=cfg.settings['ae_optimizer_name'],
                  lr=1e-3,
                   n_epochs = 2 , #100,
                   lr_milestones=(100,),
                   batch_size=100, 
                   weight_decay=0.5e-3,  
                   device=device,
                   n_jobs_dataloader=0)
    deep_SVDD.train(dataset,
                optimizer_name=cfg.settings['optimizer_name'],
                lr=1e-4,
                n_epochs = 2,
                lr_milestones=cfg.settings['lr_milestone'],
                batch_size=cfg.settings['batch_size'],
                weight_decay=cfg.settings['weight_decay'],
                device=device,
              n_jobs_dataloader=0)  
    #
    #st.write(labelled.head())
    lab = score(deep_SVDD, labelled.values).cpu().detach().numpy()*-1 
    lab_list.append(lab.ravel())
    #unlab = score(deep_SVDD, uf_final.iloc[:,:].values).cpu().detach().numpy()*-1
    #unlab_list.append(unlab.ravel())
    #st.write(labelled.head())
    return lab_list#, unlab_list

def plot_scores():
    lab_list = train_and_score()
    lab = np.mean(lab_list, 0) 
    y_scaler1 = MinMaxScaler()
    lab= y_scaler1.fit_transform(lab.reshape(-1,1))
    lab=pd.DataFrame(lab, columns=['train_score'])
    lab.describe()
    fig = plt.figure(figsize=(4,3))
    #fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    _, bins , _ = plt.hist(lab.train_score.values, bins=50, ec='k', histtype='bar', density=True, alpha=1, color='#feb308', label='Labelled Dataset Scores')
    plt.grid(False)
    plt.xlim(0.1, 1.0)
    plt.xlabel('Deep One Class Scores', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    major_ticks_x = np.arange(0.1, 1.001, 0.2)
    minor_ticks_x = np.arange(0.1 ,1.001, 0.1)
    major_ticks_y = np.arange(0, 6, 2)
    minor_ticks_y = np.arange(0, 6, 1)
    ax.tick_params(axis = 'both', which='both', width=2)
    ax.tick_params(axis = 'both', which='major', length=12)
    ax.tick_params(axis = 'both', which='minor', length=8, color='black')
    ax.tick_params(axis = 'both', which='both' , bottom=True, top=True, left=True, right=True, direction="in")
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    plt.legend(prop={'size': 10}, loc=[0.05, 0.78])
    st.pyplot(fig)
    


class Pairs_Dataset(TorchvisionDataset):
    def __init__(self, root: str, train_idx=None, test_idx=None, data=None):
        super().__init__(root)
        ## Loading the train set
        self.train_set = Pairs(root=self.root, train=True, data=data)
        if train_idx is not None:
          self.train_set = Subset(self.train_set, train_idx)
        ## Loading the test set
        self.test_set = Pairs(root=self.root, train=False, data=data)
        if test_idx is not None:
            self.test_set = Subset(self.test_set, test_idx)

class Pairs(Dataset):
    def __init__(self, root, train, data=None):
        super(Pairs, self).__init__()
        self.train = train
        if data is None:
          self.data=labelled.values.astype('f')
        else:
          self.data =  data.values.astype('f')
        self.labels = np.zeros(self.data.shape[0])       
    # This is used to return a single datapoint. A requirement from pytorch
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index
    # For Pytorch to know how many datapoints are in the dataset
    def __len__(self):
        return len(self.data)
        return self.decoder(self.encoder(x))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class PairsAutoEncoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=100, out_features=500), nn.LeakyReLU(),
        nn.Linear(in_features=500, out_features=1000),nn.LeakyReLU(),
        nn.Linear(in_features=1000, out_features=1613),nn.LeakyReLU(),         
        nn.Linear(in_features=1613, out_features=3226), nn.Sigmoid())
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

def build_autoencoder(net_name):
    return PairsAutoEncoder()

def build_network(net_name):  
    return PairsEncoder()