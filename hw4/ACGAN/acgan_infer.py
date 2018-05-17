import numpy as np
import matplotlib.pyplot as plt
from acgan_model import Generator, Discriminator
from torchvision.utils import save_image
import sys
import os
import torch
import torch._utils
from torch.autograd import Variable
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

torch.manual_seed(2)
# output_path = '/home/lilioo826/hw4_output/'
output_path = sys.argv[1]

# fig3_2
loss_G = np.load('ACGAN/loss_G_open.npy')
loss_D = np.load('ACGAN/loss_D_open.npy')
Dx = np.load('ACGAN/Dx_open.npy')
DG1 = np.load('ACGAN/DG1_open.npy')
Dacc_real = np.load('ACGAN/D_acc_real.npy')
Dacc_fake = np.load('ACGAN/D_acc_fake.npy')
plt.figure(figsize=(40,10))
plt.subplot(131)
plt.plot(loss_G[::10], label='loss G')
plt.plot(loss_D[::10], label='loss D')
plt.legend()
plt.title('loss G and loss D')
plt.subplot(132)
plt.plot(Dx[::10], label='Real')
plt.plot(DG1[::10], label='Fake')
plt.legend()
plt.title('mean of output of Discriminator')
plt.subplot(133)
plt.plot(Dacc_real[::10], label='Real')
plt.plot(Dacc_fake[::10], label='Fake')
plt.legend()
plt.title('Accuracy of Discriminator')
plt.savefig(output_path+'/fig3_2.jpg')

# fig3_3
netG = Generator(64, 256)
netG.load_state_dict(torch.load('ACGAN/acgan_netG.pth'))
fix_noise = Variable(torch.randn(10, 256, 1, 1))
fix_attr0 = Variable(torch.zeros(10, 1, 1, 1))
fix_attr1 = Variable(torch.ones(10, 1, 1, 1))
fake_0 = netG(fix_noise, fix_attr0)
fake_1 = netG(fix_noise, fix_attr1)
compare = torch.cat((fake_0, fake_1))
save_image(compare.data, output_path+'/fig3_3.jpg', nrow=10, normalize=True)