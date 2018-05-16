import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, d, kl_coef):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,stride=2),
            # nn.Conv2d(d, d, kernel_size=2, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(d, d, kernel_size=1, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
            # nn.ConvTranspose2d(d//2, 3, kernel_size=4, stride=2, padding=0, bias=False, output_padding=1),
        )
        self.d = d
        self.f = 4
        self.kl_coef = kl_coef
        self.fc11 = nn.Linear(16384, 1024)
        self.fc12 = nn.Linear(16384, 1024)
        self.fc2  = nn.Linear(1024, 16384)
        # self.fc21 = nn.Linear(d*self.f**2, d*self.f**2)
        # self.fc22 = nn.Linear(d*self.f**2, d*self.f**2)
    
    def encode(self, x):
        h1 = self.encoder(x)
        # print(h1.size())
        h1 = h1.view(h1.size(0), -1)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, 256, 8, 8)
        h3 = self.decoder(z)
        # print(z.size())
        # print(h3.size())
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x, size_average=True)
        batch_size = x.size(0)
        self.kl_loss =  -0.5 * torch.sum(1 +logvar-mu.pow(2)-logvar.exp())
        # self.kl_loss /= batch_size*3*self.d**2
        return self.mse + self.kl_coef*self.kl_loss
    
    def latest_loss(self):
        return self.mse, self.kl_loss
        