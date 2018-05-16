from vae_dataloader import FaceDataset
from torchvision import transforms
from torchvision.utils import save_image
from vae_model import VAE
from torch.autograd import Variable
from torch import optim
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(2)
place = '119'
if place == 'azure':
    data_path = '/home/hung/DLCV2018SPRING/hw4/hw4_data/'
else:
    data_path = '/home/lilioo826/hw4_data/'
train_faceDataset = FaceDataset(data_path+'train', data_path+'train.csv', transforms.ToTensor())
train_dataloader = DataLoader(train_faceDataset, batch_size=20, num_workers=1)


cuda = True
model = VAE(64, 1e-6)
# print(model)
if cuda:
    model.cuda()
# summary(model, (3,64,64))
# exit()

epoch_num = 100
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

klds = []
mses = []
for epoch in range(epoch_num):
    print('epoch {}'.format(epoch+1))
    epoch_kld = 0
    epoch_mse = 0
    epoch_loss = 0
    for batch_idx, (data, label) in enumerate(train_dataloader):
        if cuda:
            data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(data, recon_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        batch_kld, batch_mse = model.latest_loss()
        epoch_kld += batch_kld.item()/train_dataloader.batch_size
        epoch_mse += batch_mse.item()/train_dataloader.batch_size
        epoch_loss += loss.item()/train_dataloader.batch_size
    klds.append(epoch_kld)
    mses.append(epoch_mse)
    print('KLD: {}'.format(epoch_kld))
    print('MSE: {}'.format(epoch_mse))
    print('Loss: {}'.format(epoch_loss))
    # model.save_state_dict('model/epoch_{}.pt'.format(epoch))
    if epoch+1 % 10 == 0:
        torch.save(model.state_dict(), 'model/epoch_-6_{}.pt'.format(epoch+1))
        compare = torch.cat([data[:8], recon_batch[:8]])
        save_image(compare.data.cpu(), 'img_output/epoch_-6_{}.png'.format(epoch+1), nrow=8, normalize=True)
    np.save('kld_loss-7.npy', klds)
    np.save('mse_loss-7.npy', mses)
