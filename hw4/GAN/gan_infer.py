import numpy as np
import matplotlib.pyplot as plt
from gan_model import Generator, Discriminator
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
# fig2_2
loss_G = np.load('GAN/loss_G.npy')
loss_D = np.load('GAN/loss_D.npy')
Dx = np.load('GAN/Dx.npy')
DG1 = np.load('GAN/DG1.npy')
DG2 = np.load('GAN/DG2.npy')

# x = np.arange(1206)
plt.figure(figsize=(30,10))
plt.subplot(121)
plt.plot(loss_G[::10], label='loss G')
plt.plot(loss_D[::10], label='loss D')
plt.legend()
plt.title('loss G and loss D')
# plt.plot(loss_D)
plt.subplot(122)
# plt.plot(x, Dx, 'r', x, DG1, 'b')
plt.plot(Dx[::10], label='Real')
plt.plot(DG1[::10], label='Fake')
plt.legend()
plt.title('mean of output of Discriminator')
plt.savefig(output_path+'/fig2_2.jpg')

# fig2_3
netG = Generator(64, 256)
netG.load_state_dict(torch.load('GAN/gan_netG.pth'))
ran_vec = Variable(torch.randn(32, 256, 1, 1))
fake = netG(ran_vec)
save_image(fake.data, output_path+'/fig2_3.jpg', nrow=8, normalize=True)