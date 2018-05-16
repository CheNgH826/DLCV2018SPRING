import torch
from vae_model import VAE
from torchvision.utils import save_image
from vae_dataloader import FaceDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, no_grad
from sklearn.manifold import TSNE
import sys

torch.manual_seed(2)
cuda = False

place = '119'
# if place == 'azure':
#     data_path = '/home/hung/DLCV2018SPRING/hw4/hw4_data/'
# else:
#     data_path = '/home/lilioo826/hw4_data/'
# output_path = '/home/lilioo826/hw4_output/'
data_path = sys.argv[1]
output_path = sys.argv[2]

test_faceDataset = FaceDataset(data_path+'test/', data_path+'test.csv', transforms.ToTensor())
test_dataloader = DataLoader(test_faceDataset, batch_size=8, num_workers=1)

# fig1_2

kld_history = np.load('VAE/kld_loss.npy')
mse_history = np.load('VAE/mse_loss.npy')
# print(kld_history)
# print(mse_history)
plt.figure(figsize=(40, 10))
plt.subplot(121)
plt.title('KLD loss')
plt.plot(kld_history)
plt.subplot(122)
plt.title('MSE loss')
plt.plot(mse_history)
plt.savefig(os.path.join(output_path, 'fig1_2.jpg'))


model = VAE(64, 1e-5)
model.load_state_dict(torch.load('VAE/vae_state_model.pth'))
if cuda:
    model = model.cuda()
# print(model)

mse = 0
data_for_tsne = []
label_for_tsne = []
for (data, label) in test_dataloader:
    # print(data.size())
    if cuda:
        data = no_grad(data.cuda(), requires_grad=False)
    recon_img, mu, logvar = model(data)
    loss = model.loss_function(data, recon_img, mu, logvar)
    mse += torch.sum(model.latest_loss()[0])
    if len(data_for_tsne) < 50:
        data_for_tsne.append(data)
        label_for_tsne.append(label)
print(mse)

# fig1_3
recon_imgs = []
oirgin_imgs = []
for i in range(10):
    oirgin_imgs.append(test_faceDataset[i][0])
# for img in test_faceDataset[:10]:
for img in oirgin_imgs:
    out_img = model(img.unsqueeze_(0))[0]
    recon_imgs.append(out_img)
oirgin_imgs = torch.cat(oirgin_imgs)
recon_imgs = torch.cat(recon_imgs)
compare = torch.cat((oirgin_imgs, recon_imgs), 0)
save_image(compare.data.cpu(), os.path.join(output_path, 'fig1_3.jpg'), nrow=10, normalize=True)

# fig1_4
imgs = []
for i in range(32):
    z = torch.randn(1024)
    out_img = model.decode(z)
    imgs.append(out_img)
imgs = torch.cat(imgs)
save_image(imgs.data.cpu(), os.path.join(output_path, 'fig1_4.jpg'), nrow=8, normalize=True)

# fig1_5
data_for_tsne = torch.cat(data_for_tsne)
label_for_tsne = torch.cat(label_for_tsne).numpy()
mu, logvar = model.encode(data_for_tsne)
latent_code = mu.cpu().data.numpy()
latent_emmbedded = TSNE(random_state=2).fit_transform(latent_code)
plt.figure()
# for i in [0,1]:
#     if i:
#         gender = 'Male'
#     else:
#         gender = 'Female'
#     xy = latent_emmbedded[label_for_tsne==i]
#     plt.scatter(xy[:,0], xy[:,1], c=i, label=gender)
plt.scatter(latent_emmbedded[:,0], latent_emmbedded[:,1], c=label_for_tsne)
# plt.legend()
plt.savefig(os.path.join(output_path, 'fig1_5.jpg'))
