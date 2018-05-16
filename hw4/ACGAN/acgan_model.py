import torch
from torch import nn

# d = 64
# latent_size = 256
class Generator(nn.Module):
    def __init__(self, d, latent_size):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size+1, d*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d*8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(d, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x, label):
        x = torch.cat((x, label), 1)
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, d):
        super(Discriminator, self).__init__()
        self.smallConv = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.8),
            nn.Conv2d(d, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.8),
            nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.8),
            nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.discrim = nn.Sequential(
            # nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(d*8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*8, 1, 4, 1, 0, bias=False),
            # nn.Dropout2d(0.5),
            nn.Sigmoid()
        )
            # nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(d*8),
            # nn.LeakyReLU(0.2, inplace=True),
        self.classify = nn.Sequential(
            nn.Conv2d(d*8, 1, 4, 1, 0, bias=False),
            # nn.Linear(512*4*4, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        mid = self.smallConv(x)
        realfake = self.discrim(mid)#.view(-1, 1).squeeze(1)
        # label = self.classify(mid.view(-1, 512*4*4))#.view(-1, 1).squeeze(1)
        label = self.classify(mid)#.view(-1, 1).squeeze(1)
        return realfake, label
