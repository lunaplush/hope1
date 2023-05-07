import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import seaborn as sns

sns.set(style='darkgrid', font_scale=1.2)
master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
df_small_noise_url = master_url_root + df_small_noise_url_suffix
df_small_noise = pd.read_csv(
    df_small_noise_url, parse_dates=True, index_col="timestamp"
)

df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
)

training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
print(training_mean, training_std)
df_training_value = (df_small_noise - training_mean) / training_std

TIME_STEPS = 288


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    print(len(values))
    for i in range(len(values) - time_steps):
        output.append(values[i: (i + time_steps)])
    return torch.FloatTensor(output).permute(0, 2, 1)


X = create_sequences(df_training_value.values)

x_train, x_val = train_test_split(X, test_size=0.9, shuffle=False)
print("Training input shape: ", x_train.shape)

train_loader = torch.utils.data.DataLoader(x_train, batch_size=4)
val_loader = torch.utils.data.DataLoader(x_val, batch_size=4)

TIME_STEPS = 288


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    print(len(values))
    for i in range(len(values) - time_steps):
        output.append(values[i: (i + time_steps)])
    output= np.array(output)
    return torch.FloatTensor(output).permute(0, 2, 1)


X = create_sequences(df_training_value.values)

x_train, x_val = train_test_split(X, test_size=0.9, shuffle=False)
print("Training input shape: ", x_train.shape)

train_loader = torch.utils.data.DataLoader(x_train, batch_size=4)
val_loader = torch.utils.data.DataLoader(x_val, batch_size=4)

class Autoencoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
        nn.ReLU(),
        nn.Conv1d(32, 16, kernel_size=7, stride=1, padding=3),
        nn.ReLU(),
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose1d(16, 16, kernel_size=7, stride=1, padding=3),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.ConvTranspose1d(16, 32, kernel_size=7, stride=1, padding=3),
        nn.ReLU(),
        nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3),
    )

  def forward(self, sample):
    latent = self.encoder(sample)
    reconstructed = self.decoder(latent)
    return reconstructed

n_epochs = 2
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = F.mse_loss
train_losses = []
val_losses = []

for epoch in tqdm_notebook(range(n_epochs)):
  model.train()
  train_losses_per_epoch = []
  for i, X_batch in enumerate(train_loader):
      optimizer.zero_grad()
      reconstructed = model(X_batch.to(device))
      loss = loss_fn(reconstructed, X_batch.to(device))
      loss.backward()
      optimizer.step()
      train_losses_per_epoch.append(loss.item())

  train_losses.append(np.mean(train_losses_per_epoch))

  model.eval()
  val_losses_per_epoch = []
  with torch.no_grad():
      for X_batch in val_loader:
          reconstructed = model(X_batch.to(device))
          loss = loss_fn(reconstructed, X_batch.to(device))
          val_losses_per_epoch.append(loss.item())

  val_losses.append(np.mean(val_losses_per_epoch))
