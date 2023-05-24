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
from autoencoder_net import AutoencoderConv


from fetch_dataset import fetch_dataset
# FETCH_DATA = True
#Используется только для первого запуска. Данные  считываются  с диска  и сохраняются в виде объектов  в файлах
#  main_data.df and main_attrs.df
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

device = ("cuda:0" if torch.cuda.is_available() else "cpu")



def train(model, loss_fn, epochs, epoch_start, X_tr, X_vl,  scheduler, optimizator):
    analysis_dict = {"loss_train": [], "loss_val": []}
    next_epoch_num = epoch_start
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (next_epoch_num + 1, epoch_start + epochs))
        for x in X_tr:
            loss_step = 0
            model.train()
            optimizator.zero_grad()
            x_rec, x_lat = encoder(x)
            loss = loss_fn(x, x_rec)
             #print(loss)
            #print(x_rec[0][0])

            loss.backward()
            optimizator.step()
            if loss_step == 0:
                loss_step = loss
            else:
                loss_step = (loss_step + loss) / 2
        # print(x_rec[0][0][0], x[0][0][0])
        analysis_dict["loss_train"].append(loss_step.detach().numpy())
        print(f"on {epoch + epoch_start + 1} / {epochs + epoch_start} loss: {loss}")
        x_vl = iter(X_vl).next()
        x_vl_rec, x_vl_lat = model(x_vl)
        loss_val = loss_fn(x_vl, x_vl_rec)
        analysis_dict["loss_val"].append(loss_val.detach().numpy())


        fig, ax = plt.subplots(2, 6, figsize=(14, 6))
        for i in range(6):
            ax[0][i].imshow(x_vl[i].detach().numpy())
            ax[1][i].imshow(x_vl_rec[i].detach().numpy())
            ax[0][i].axis('off')
            ax[1][i].axis("off")


        fig.suptitle('%d / %d - loss: %f' % (next_epoch_num + 1, epoch_start + epochs, loss_val))
        fig.savefig(str(next_epoch_num+1)+".jpg")
        next_epoch_num += 1
        scheduler.step()

    return pd.DataFrame(analysis_dict)


encoder = AutoencoderConv(latent_dim=32).to(device=device)
#1 - Установить FIRST = TRUE -если сеть только начинает обучатся
FIRST = False

net_name = "ae_conv_32"
batch_size = 16

if FIRST:
    epoch_start = 0
    if not os.path.exists(net_name + "_epochs"):
        os.mkdir(net_name + "_epochs")

else:
    f = open(net_name+"_epoch.num", "rb")
    epoch_start = pickle.load(f)
    f.close()
    encoder.load_state_dict(torch.load(net_name + ".net"))



#optimizator = optim.AdamW(encoder.parameters(), lr=0.0001, weight_decay=0.05)
optimizator = optim.Adam(encoder.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizator, milestones=[15,40], gamma=0.2)
loss_fn = F.mse_loss

train_losses = []
test_loses = []
epochs = 20


X_train, X_val = train_test_split(np.array(data, np.float32), train_size=0.8, shuffle=True, random_state=100)
X_tr = DataLoader(X_train, batch_size=batch_size)
X_vl = DataLoader(X_val, batch_size=batch_size)
work_path = os.getcwd()
os.chdir(net_name+"_epochs")
analysis_data = train(encoder, loss_fn, epochs, epoch_start, X_tr, X_vl, scheduler, optimizator)
os.chdir(work_path)
torch.save(encoder.state_dict(), net_name + ".net")


f = open(net_name+"_epoch.num", "wb")
epoch_next = epoch_start + epochs
pickle.dump(epoch_next, f)
f.close()


if FIRST:
    analysis_data.to_csv(net_name + ".csv")
else:
    analysis_data.to_csv(net_name+".csv", mode="a", header=False)
