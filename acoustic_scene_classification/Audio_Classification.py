#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import librosa
import librosa.display
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdm
import os
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[82]:


df = pd.read_pickle('/home/sauravpathak/acoustic_scene_classification_small/train_split_df.pkl')
np.random.shuffle(df.values)
print('df loaded')

# In[83]:


dataset_size = len(df)


# In[84]:


def get_melspectrogram_db(file_path, sr = 48000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=24000, top_db=80):
    wav,sr = librosa.load(file_path, sr = sr)
    if wav.shape[0]<2*sr:
        wav=np.pad(wav,int(np.ceil((2*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:2*sr]
    spec=librosa.feature.melspectrogram(wav,sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db


# In[85]:


spec = get_melspectrogram_db(df.iloc[0,1])
print(spec.shape)
del spec


# In[86]:


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 1 * (spec_norm - spec_min) / (spec_max - spec_min)
    return spec_scaled


# In[87]:


class AudioData(Dataset):
    def __init__(self, df, out_col):
        self.df = df
        self.data = []
        self.labels = []
        self.c2i={}
        self.i2c={}
        self.categories = sorted(df[out_col].unique())
        for i, category in enumerate(self.categories):
            self.c2i[category]=i
            self.i2c[i]=category
        for ind in range(len(df)):
            row = df.iloc[ind]
            file_path = df.iloc[ind,1]
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
            self.labels.append(self.c2i[row['label']])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# In[88]:


def data_loader(df, batch_size):
    train_df = df.iloc[:int(np.floor(dataset_size)*0.8)]
    valid_df = df.iloc[int(np.floor(dataset_size)*0.8):]
    train_data = AudioData(train_df, 'label')
    valid_data = AudioData(valid_df, 'label')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print('train_loader loaded')
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    print('valid_loader loaded')
    return train_loader, valid_loader


# In[89]:

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

def model_init():
    resnet_model = resnet34(pretrained=False)
    resnet_model.fc = nn.Linear(512,7)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model = resnet_model.to(device)
    return resnet_model


# In[92]:


from torch.optim.lr_scheduler import StepLR


# In[93]:


epochs = 100


# In[94]:


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


# In[95]:


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler):
    train_acc = 0
    valid_acc = 0
    train_batch_losses=[]
    valid_batch_losses=[]
    trace_y = []
    trace_yhat = []
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)  
            loss.backward()
            if epoch == epochs:
                train_batch_losses.append(loss.item())
                trace_y.append(y.cpu().detach().numpy())
                trace_yhat.append(y_hat.cpu().detach().numpy())  
            optimizer.step()
        scheduler.step()
        if epoch == epochs:
            trace_y = np.concatenate(trace_y)
            trace_yhat = np.concatenate(trace_yhat)
            train_acc = np.mean(trace_yhat.argmax(axis=1)==trace_y)

        if epoch == epochs:
            with torch.no_grad():
                model.eval()
                trace_y = []
                trace_yhat = []
                for i, data in enumerate(valid_loader):
                    x, y = data
                    x = x.to(device, dtype=torch.float32)
                    y = y.to(device, dtype=torch.long)
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)
                    trace_y.append(y.cpu().detach().numpy())
                    trace_yhat.append(y_hat.cpu().detach().numpy())      
                    valid_batch_losses.append(loss.item())
                trace_y = np.concatenate(trace_y)
                trace_yhat = np.concatenate(trace_yhat)
                valid_acc = np.mean(trace_yhat.argmax(axis=1)==trace_y)
    return train_acc, valid_acc, np.mean(train_batch_losses), np.mean(valid_batch_losses)


# In[96]:


def grid_model(df, epochs, model_init = model_init, init_weights = init_weights, params = {'weight_decay': 0.001, 'lr': 0.01, 'batch_size': 16}):
    model = model_init()
    model.apply(init_weights)
    train_loader, valid_loader = data_loader(df, params['batch_size'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.2)
    train_acc, valid_acc, train_loss, valid_loss = train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler)
    return train_acc, valid_acc, train_loss, valid_loss


# In[97]:


from sklearn.model_selection import ParameterGrid


# In[98]:


param_grid = {'weight_decay': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'lr': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'batch_size': [4, 8, 16, 32]}


# In[99]:


parameter_grid = ParameterGrid(param_grid)


# In[100]:


train_accuracy = []
valid_accuracy = []
train_losses = []
valid_losses = []


# In[101]:


def grid_search(epochs, parameter_grid, df):
    for grid in tqdm(parameter_grid):
        train_acc, valid_acc, train_loss, valid_loss = grid_model(df, epochs, params = grid)
        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)


# In[ ]:


grid_search(epochs, parameter_grid, df)
np.save('/home/sauravpathak/acoustic_scene_classification_small/train_accuracy.npy', train_accuracy)
np.save('/home/sauravpathak/acoustic_scene_classification_small/valid_accuracy.npy', valid_accuracy)
np.save('/home/sauravpathak/acoustic_scene_classification_small/train_losses.npy', train_losses)
np.save('/home/sauravpathak/acoustic_scene_classification_small/valid_losses.npy', valid_losses)

