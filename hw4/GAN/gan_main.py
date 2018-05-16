from torch.utils.data import DataLoader, ConcatDataset
from gan_dataloader import FaceDataset
from torchvision import transforms
from gan_model import Generator, Discriminator
from torch import nn, optim
from torchvision.utils import save_image
from torchsummary import summary
import torch
import numpy as np

torch.manual_seed(2)
cuda = True
place = '119'
if place == 'azure':
    data_path = '/home/hung/DLCV2018SPRING/hw4/hw4_data/'
else:
    data_path = '/home/lilioo826/hw4_data/'
epoch_num = 50
batch_size = 64
d = 64
latent_size = 256
mode = 'lat256'

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)

train_faceDataset = FaceDataset(data_path+'train', data_path+'train.csv', transform2)
test_faceDataset = FaceDataset(data_path+'test', data_path+'test.csv', transform2)
train_dataloader = DataLoader(ConcatDataset([train_faceDataset, test_faceDataset]), batch_size=batch_size, num_workers=1)

netG = Generator(d, latent_size)
netD = Discriminator(d)
if cuda:
    netG = netG.cuda()
    netD = netD.cuda()
#print(netG)
#summary(netG, (1, 128))
# print(netD)
# summary(netD, (3, 64, 64))
# exit()

criterion = nn.BCELoss()

optimizerG = optim.Adam(netG.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.002, betas=(0.5, 0.999))

fix_noise = torch.randn(batch_size, latent_size, 1, 1).cuda()
lossG = []
lossD = []
Dx = []
DG1 = []
DG2 = []
for epoch in range(epoch_num):
    for i, data in enumerate(train_dataloader):
        # Update D network
        # train with real
        if cuda:
            data = data.cuda()
        netD.zero_grad()
        label = torch.full((data.size(0), ), 1).cuda()
        output = netD(data)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(data.size(0), latent_size, 1, 1).cuda()
        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # Update G network
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
            %(epoch, epoch_num, i, len(train_dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i%10==0:
            lossD.append(errD.item())
            lossG.append(errG.item())
            Dx.append(D_x)
            DG1.append(D_G_z1)
            DG2.append(D_G_z2)
        
    save_image(data[:32], 'img_output/real_samples_%s.png'%mode, normalize=True)
    fake = netG(fix_noise[:32])
    save_image(fake.detach(), 'img_output/fake_samples_%s_%d.png'%(mode, epoch), normalize=True)
    np.save('loss_G_%s.npy'%mode, np.array(lossG))
    np.save('loss_D_%s.npy'%mode, np.array(lossD))
    np.save('Dx_%s.npy'%mode, np.array(Dx))
    np.save('DG1_%s.npy'%mode, np.array(DG1))
    np.save('DG2_%s.npy'%mode, np.array(DG2))
    if (epoch+1)%3 == 0:
        torch.save(netG.state_dict(), 'model/netG_epoch_%s_%d.pth' %(mode, epoch))
        torch.save(netD.state_dict(), 'model/netD_epoch_%s_%d.pth' %(mode, epoch))
