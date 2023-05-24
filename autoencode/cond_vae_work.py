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
from torchvision.utils import save_image
from sklearn.preprocessing import OneHotEncoder
from cond_vae_net import CVAE


from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)




latent_dim = 32
lr = 1e-4
ohe = OneHotEncoder(max_categories=10, sparse_output=False).fit(torch.LongTensor(np.array(range(10)).reshape(10, 1)))

model = CVAE(784, latent_dim, ohe).to(device)


# Найдем мю и логсигма для цифы 1
def generate_numbers(number):
    mu = 0
    # print(mu)
    logsigma = 1
    # print(logsigma)
    covs = np.zeros((latent_dim, latent_dim))
    for i in range(latent_dim):
        covs[i][i] = 1
    z = np.random.multivariate_normal(mu, covs, latent_dim)
    numbers = np.ones((latent_dim, 1)) * number
    print(numbers.shape)
    output = model.decode(torch.FloatTensor(z), torch.LongTensor(numbers))
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    for i in range(5):
        for j in range(2):
            # ax[0][i].imshow(x_vl[i].detach().numpy())
            # ax[1][i].imshow(x_vl_rec[i].detach().numpy())

            ax[j][i].imshow(
                torch.permute(torch.clip(output[i + 5 * j], 0, 1).view(1, 28, 28), (1, 2, 0)).detach().numpy())
            ax[j][i].axis('off')

    fig.suptitle(f"сегенерированное число  {number}")
    fig.savefig("tmp1.jpg")
generate_numbers(5)