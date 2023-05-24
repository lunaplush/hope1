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
from vae_net import VAE
import torch.nn.functional as F


from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
net_name ="vae1"
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

latent_dim = 2

model = VAE(784, latent_dim).to(device)


# z = np.array([np.random.normal(0, 1, 100) for i in range(10)])
# print(z.shape)
# output = model.decode(torch.FloatTensor(z))
# fig, ax = plt.subplots(2, 5, figsize=(14, 6))
# for i in range(5):
#     for j in range(2):
#         # ax[0][i].imshow(x_vl[i].detach().numpy())
#         # ax[1][i].imshow(x_vl_rec[i].detach().numpy())
#         ax[j][i].imshow(torch.clip(output[i + 5 * j], 0, 1).detach().numpy())
#         ax[j][i].axis('off')
#
# fig.suptitle("from latent")
# fig.savefig("tmp1.jpg")



