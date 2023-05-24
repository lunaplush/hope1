from torch import nn
import torch.nn.functional as F
import torch
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc21 = nn.Linear(500, latent_dim)  # fc21 for mean of Z
        self.fc22 = nn.Linear(500, latent_dim)  # fc22 for log variance of Z
        self.fc3 = nn.Linear(latent_dim, 500)
        self.fc4 = nn.Linear(500, input_dim)

    def encode(self, x):
        x1 = F.relu(self.fc1(x))
        mu = self.fc21(x1)
        logsigma = self.fc22(x1)
        return mu, logsigma

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(0.5 * logsigma)
            eps = torch.rand_like(std)
            return mu + eps * std
        else:
            # на инференсе возвращаем не случайный вектор из нормального распределения, а центральный -- mu.
            # на инференсе выход автоэнкодера должен быть детерминирован.
            return mu

    def decode(self, z):
        x = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(x))

    def forward(self, x):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]

        x = x.view(-1, self.input_dim)
        mu, logsigma = self.encode(x)
        z = self.gaussian_sampler(mu, logsigma)
        return self.decode(z), mu, logsigma