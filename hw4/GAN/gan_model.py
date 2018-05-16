import torch
from torch import nn

# d = 64
# latent_size = 256
class Generator(nn.Module):
    def __init__(self, d, latent_size):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, d*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            nn.ConvTranspose2d(d, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, d):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.encoder(x).view(-1, 1).squeeze(1)

# netG = generator()
