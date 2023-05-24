import numpy as np
from torch.autograd import Variable
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import pandas as pd
import skimage.io
from skimage.transform import resize
import pickle
from sklearn.model_selection import train_test_split
from autoencoder_net import Autoencoder
net_model ="ae32d"

from fetch_dataset import fetch_dataset
FETCH_DATASET = False
if FETCH_DATASET:
    data, attrs = fetch_dataset()
    f = open("main_data.df", "wb")
    pickle.dump(data, f)
    f.close()
    f = open("main_attrs.df", "wb")
    pickle.dump(attrs, f)
    f.close()
else:
    f = open("main_data.df", "rb")
    data = pickle.load(f)
    f.close()
    f = open("main_attrs.df", "rb")
    attrs = pickle.load(f)
    f.close()
latent_dim = 32
model = Autoencoder((64, 64, 3), latent_dim=latent_dim)
model.load_state_dict(torch.load(net_model+".net"))


loss_fn = F.mse_loss

train_losses = []
test_loses = []
epochs = 15
batch_size = 16

X_train, X_val = train_test_split(np.array(data, np.float32), train_size=0.8, shuffle=True, random_state=100)
X_tr = DataLoader(X_train, batch_size=batch_size, drop_last=True)
X_vl = DataLoader(X_val, batch_size=batch_size)

GET_FROM_LATENT = True
if GET_FROM_LATENT:

    latents = np.zeros((len(X_tr)*batch_size, latent_dim))
    i=0
    for x in X_tr:
        x_rec, x_lat = model(x)
        latents[i*batch_size:(i+1)*batch_size] = x_lat.detach().numpy()
        i += 1


    means = latents.mean(axis=0)
    std = latents.std(axis=0)
    var = latents.var(axis=0)
    covs = np.zeros((latent_dim, latent_dim))
    for i in range(latent_dim):
        covs[i][i] = var[i]
    # сгенерируем 25 рандомных векторов размера latent_space
    # z = np.random.randn(25, latent_dim)
    z = np.random.multivariate_normal(means, covs, latent_dim)
    output = model.decode(torch.FloatTensor(z))
    fig, ax = plt.subplots(5, 5, figsize=(14, 14))
    for i in range(5):
        for j in range(5):
            # ax[0][i].imshow(x_vl[i].detach().numpy())
            # ax[1][i].imshow(x_vl_rec[i].detach().numpy())
            ax[j][i].imshow(torch.clip(output[i + 5 * j], 0, 1).detach().numpy())
            ax[j][i].axis('off')

    fig.suptitle("from latent")
    fig.savefig(net_model+"from_latent_vector.jpg")

DO_SMILE = True
if DO_SMILE:
    smile = data[attrs.Smiling > 2]
    no_smile = data[attrs.Smiling < -2]

    latents_smile = np.zeros((len(smile), latent_dim))
    latents_no_smile = np.zeros((len(no_smile), latent_dim))
    i = 0
    for i in range(len(smile)):
        x_rec, x_lat = model(torch.FloatTensor(smile[i]).view(1, -1))
        latents_smile[i] = x_lat[0].detach().numpy()
        i += 1
    i = 0
    for i in range(len(no_smile)):
        x_rec, x_lat = model(torch.FloatTensor(no_smile[i]).view(1, -1))
        latents_no_smile[i] = x_lat[0].detach().numpy()
        i += 1


    smile_means = latents_smile.mean(axis=0)
    no_smile_means = latents_no_smile.mean(axis=0)
    smile_vector = smile_means - no_smile_means

    fig, ax = plt.subplots(2, 6)
    x_rec, x_lat = model(torch.FloatTensor(no_smile[0:12]))
    x_lat_smile = x_lat.sub(torch.FloatTensor(smile_vector))
    x_rec = model.decode(x_lat_smile)
    for i in range(6):
        #ax[0][i].imshow(torch.clip(x_rec[i], 0, 1).detach().numpy())
        #ax[1][i].imshow(torch.clip(x_rec[6+i], 0, 1).detach().numpy())
        ax[0][i].imshow(no_smile[i])
        ax[1][i].imshow(torch.clip(x_rec[i], 0, 1).detach().numpy())
        ax[0][i].axis("off")
        ax[1][i].axis("off")

    fig.savefig("tmp3_2.jpg")


    print(smile_means)
    print(no_smile_means)
    print(smile_means-no_smile_means)
