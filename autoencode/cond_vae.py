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
from cond_vae_net import CVAE
from sklearn.preprocessing import OneHotEncoder


from torchvision import datasets, transforms



LOG_INTERVAL = 100
PRR = True
Z1_RANGE = 2
Z2_RANGE = 2
Z1_INTERVAL = 0.2
Z2_INTERVAL = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)




def KL_divergence(mu, logsigma):
    """
    часть функции потерь, которая отвечает за "близость" латентных представлений разных людей
    """
    loss = 0.5 * torch.sum(mu.pow(2) + logsigma.exp() - logsigma - 1)
    return loss

def log_likelihood(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции (как mse в обычном autoencoder)
    """
    loss = F.binary_cross_entropy(reconstruction, x.view(-1, 784), reduction='sum')
    return loss

def loss_vae(reconstruction, x, mu, logsigma):
    BCE = log_likelihood(x, reconstruction)
    KLD = KL_divergence(mu, logsigma)
    return BCE + KLD

# --- train and test --- #
def train(model, loss_fn, epochs, epoch_start, train_loader, test_loader, scheduler, ohe):

    analysis_dict = {"loss_train": [], "loss_val": []}
    next_epoch_num = epoch_start
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (next_epoch_num + 1, epoch_start + epochs))
        for x, y in train_loader:
            loss_step = 0
            model.train()
            optimizator.zero_grad()
            x_rec, mu, logsigma = model(x.to(device), y.to(device))
            loss = loss_fn(x_rec, x, mu, logsigma)
            loss.backward()
            optimizator.step()
            if loss_step == 0:
                loss_step = loss
            else:
                loss_step = (loss_step + loss) / 2
        analysis_dict["loss_train"].append(loss_step.detach().numpy())
        print(f"on {epoch + epoch_start + 1} / {epochs + epoch_start} loss: {loss}")
        x_vl, y_vl = iter(test_loader).next()
        x_vl_rec, x_vl_mu, x_vl_logsigma = model(x_vl, y_vl)
        loss_val = loss_fn(x_vl_rec, x_vl, x_vl_mu, x_vl_logsigma)
        analysis_dict["loss_val"].append(loss_val.detach().numpy())

        fig, ax = plt.subplots(2, 6, figsize=(14, 6))
        for i in range(6):
            ax[0][i].imshow(torch.permute(x_vl[i],(1,2,0)).detach().numpy())
            ax[1][i].imshow(torch.permute(x_vl_rec[i].view(1,28,28),(1,2,0)).detach().numpy())
            ax[0][i].axis('off')
            ax[1][i].axis("off")

        fig.suptitle('%d / %d - loss: %f' % (next_epoch_num + 1, epoch_start + epochs, loss_val))
        fig.savefig(str(next_epoch_num + 1) + ".jpg")
        next_epoch_num += 1
        scheduler.step()

    return pd.DataFrame(analysis_dict)



# --- etc. funtions --- #
def save_generated_img(image, name, epoch, nrow=8):
    if not os.path.exists('results'):
        os.makedirs('results')

    if epoch % 5 == 0:
        save_path = 'results/'+name+'_'+str(epoch)+'.png'
        save_image(image, save_path, nrow=nrow)


def sample_from_model(epoch):
    with torch.no_grad():
        # p(z) = N(0,I), this distribution is used when calculating KLD. So we can sample z from N(0,I)
        sample = torch.randn(64, latent_dim).to(device)
        sample = model.decode(sample).cpu().view(64, 1, 28, 28)
        save_generated_img(sample, 'sample', epoch)


def plot_along_axis(epoch):
    z1 = torch.arange(-Z1_RANGE, Z1_RANGE, Z1_INTERVAL).to(device)
    z2 = torch.arange(-Z2_RANGE, Z2_RANGE, Z2_INTERVAL).to(device)
    num_z1 = z1.shape[0]
    num_z2 = z2.shape[0]
    num_z = num_z1 * num_z2

    sample = torch.zeros(num_z, 2).to(device)

    for i in range(num_z1):
        for j in range(num_z2):
            idx = i * num_z2 + j
            sample[idx][0] = z1[i]
            sample[idx][1] = z2[j]

    sample = model.decode(sample).cpu().view(num_z, 1, 28, 28)
    save_generated_img(sample, 'plot_along_z1_and_z2_axis', epoch, num_z1)


FIRST = True
net_name ="cvae2"

epochs = 4
latent_dim = 32
lr = 1e-4


model = CVAE(784, latent_dim, ohe).to(device)

if FIRST:
    epoch_start = 0
    if not os.path.exists(net_name + "_epochs"):
        os.mkdir(net_name + "_epochs")

else:
    f = open(net_name+"_epoch.num", "rb")
    epoch_start = pickle.load(f)
    f.close()
    model.load_state_dict(torch.load(net_name + ".net"))



optimizator = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizator, milestones=[15], gamma=0.2)


loss_fn = loss_vae

train_losses = []
test_loses = []

work_path = os.getcwd()
os.chdir(net_name+"_epochs")
analysis_data = train(model, loss_fn, epochs, epoch_start, train_loader, test_loader, scheduler, ohe)
os.chdir(work_path)
torch.save(model.state_dict(), net_name + ".net")


f = open(net_name+"_epoch.num", "wb")
epoch_next = epoch_start + epochs
pickle.dump(epoch_next, f)
f.close()


if FIRST:
    analysis_data.to_csv(net_name + ".csv")
else:
    analysis_data.to_csv(net_name+".csv", mode="a", header=False)




# # --- main function --- #
# if __name__ == '__main__':
#     for epoch in range(1, epochs + 1):
#         train(epoch)
#         test(epoch)
#         sample_from_model(epoch)
#
#         if PRR:
#             plot_along_axis(epoch)