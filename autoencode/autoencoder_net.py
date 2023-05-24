import torch
from torch import nn
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        input_dim_flat = input_dim[0]*input_dim[1]*input_dim[2]
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim_flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim_flat),
            nn.Unflatten(1, (input_dim[0], input_dim[1], input_dim[2]))
        )

    def forward(self, sample):
        latent = self.encoder(sample)
        reconstruct = self.decoder(latent)
        return reconstruct, latent



    def decode(self, latent):
        return self.decoder(latent)

class AutoencoderConv(nn.Module):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.encoder_conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.encoder_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True, ceil_mode=True)
        self.encoder_lin = nn.Sequential(
            nn.Flatten(1), # x.view(self.shape)
            nn.Linear(2048, latent_dim, bias=True)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 2048, bias=True),
            nn.Unflatten(1, (32, 8, 8))
        )
        self.decoder_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.decoder_unconv0 = nn.Sequential(
            nn.Dropout(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.decoder_unconv1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, sample):
        x = torch.permute(sample, (0, 3, 1, 2))
        x1 = self.encoder_conv0(x)
        x11 = self.encoder_conv1(x1)
        x2, pool_indices = self.encoder_pool(x11)
        latent = self.encoder_lin(x2)
        x3 = self.decoder_lin(latent)
        x4 = self.decoder_unpool(x3, pool_indices)
        reconstracted = self.decoder_unconv1(self.decoder_unconv0(x4))
        return torch.permute(reconstracted, (0, 2, 3, 1)), latent
