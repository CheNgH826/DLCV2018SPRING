from torch.utils.data import DataLoader, ConcatDataset
from acgan_dataloader import FaceDatasetAttr
from torchvision import transforms
from acgan_model import Generator, Discriminator
from torch import nn, optim
from torchvision.utils import save_image
import torch
import numpy as np
from torchsummary import summary
from sklearn.metrics import accuracy_score

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

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)

train_faceDataset = FaceDatasetAttr(data_path+'train', data_path+'train.csv', transform2)
test_faceDataset = FaceDatasetAttr(data_path+'test', data_path+'test.csv', transform2)
train_dataloader = DataLoader(ConcatDataset([train_faceDataset, test_faceDataset]), batch_size=batch_size, num_workers=1)

netG = Generator(d, latent_size)
netD = Discriminator(d)
if cuda:
    netG = netG.cuda()
    netD = netD.cuda()
# summary(netD, (3, 64, 64))
# exit()

criterion = nn.BCELoss()

optimizerG = optim.Adam(netG.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.002, betas=(0.5, 0.999))

fix_noise = torch.randn(10, latent_size, 1, 1).cuda()
fix_attr0 = torch.zeros(10, 1, 1, 1).cuda()
fix_attr1 = torch.ones(10, 1, 1, 1).cuda()
lossG = []
lossD = []
Dx = []
DG1 = []
DG2 = []
D_acc_real = []
D_acc_fake = []
for epoch in range(epoch_num):
    for i, (data, attr) in enumerate(train_dataloader):
        # Update D network
        # train with real
        if cuda:
            data = data.cuda()
        netD.zero_grad()
        label = torch.full((data.size(0), ), 1).cuda()
        attr_label = attr.cuda()
        output, attr_pred_real = netD(data)
        errD_real = criterion(output, label) + criterion(attr_pred_real, attr_label.type(torch.FloatTensor).cuda())
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(data.size(0), latent_size, 1, 1).cuda()
        attr_noise = torch.randint(2, (data.size(0), 1, 1, 1)).cuda()
        fake = netG(noise, attr_noise)
        label.fill_(0)
        output, attr_pred = netD(fake.detach())
        errD_fake = criterion(output, label)# + criterion(attr_pred, attr_noise.type(torch.FloatTensor).cuda())
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # Update G network
        netG.zero_grad()
        label.fill_(1)
        output, attr_pred_fake = netD(fake)
        errG = criterion(output, label) + criterion(attr_pred_fake, attr_noise.type(torch.FloatTensor).cuda())
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

            attr_pred_real_np = attr_pred_real.cpu().data.numpy().reshape(-1)
            attr_pred_real_np[attr_pred_real_np>0.5] = 1
            attr_pred_real_np[attr_pred_real_np<=0.5] = 0
            attr_pred_fake_np = attr_pred_fake.cpu().data.numpy().reshape(-1)
            attr_pred_fake_np[attr_pred_fake_np>0.5] = 1
            attr_pred_fake_np[attr_pred_fake_np<=0.5] = 0
            attr_label_np = attr_label.cpu().numpy().reshape(-1)
            attr_noise_np = attr_noise.cpu().numpy().reshape(-1)
            D_acc_real.append(accuracy_score(attr_pred_real_np, attr_label_np))
            D_acc_fake.append(accuracy_score(attr_pred_fake_np, attr_noise_np))
        
    # save_image(data[:32], 'img_output/real_samples.png', normalize=True)
    fake_0 = netG(fix_noise, fix_attr0)
    fake_1 = netG(fix_noise, fix_attr1)
    compare = torch.cat((fake_0, fake_1))
    # save_image(fake_0.detach(), 'img_output/fake_samples0_%d.png'%(epoch), normalize=True)
    # save_image(fake_1.detach(), 'img_output/fake_samples1_%d.png'%(epoch), normalize=True)
    save_image(compare, 'img_output/fake_samples_%d_open.png'%(epoch), nrow=10, normalize=True)
    np.save('loss_G_open.npy', np.array(lossG))
    np.save('loss_D_open.npy', np.array(lossD))
    np.save('Dx_open.npy', np.array(Dx))
    np.save('DG1_open.npy', np.array(DG1))
    np.save('DG2_open.npy', np.array(DG2))
    np.save('D_acc_real.npy', np.array(D_acc_real))
    np.save('D_acc_fake.npy', np.array(D_acc_fake))
    torch.save(netG.state_dict(), 'model/netG_epoch_%d_open.pth' % epoch)
    torch.save(netD.state_dict(), 'model/netD_epoch_%d_open.pth' % epoch)
