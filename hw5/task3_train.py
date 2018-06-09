import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import Task2Model, Task2Model_LSTM, resnet50, transform
import torch.optim as optim
import torch.nn.functional as F
import os
from skimage import io
from tqdm import tqdm
from tensorboardX import SummaryWriter

writer = SummaryWriter('summary/task3')

frame_path = 'HW5_data/FullLengthVideos/videos/train/'
label_path = 'HW5_data/FullLengthVideos/labels/train/'
model_path = 'models/task2_best_0.4506.pth'
video_namelist = os.listdir(frame_path)
# print(len(video_namelist))

hidden_size = 512
model = Task2Model_LSTM(2048*4*2, hidden_size, 11)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
optimizer.load_state_dict(checkpoint['optimizer'])
criterion = nn.NLLLoss()

epoch_num = 100
loss_history = []
step = 1
for epoch in range(epoch_num):
    for j, video_name in enumerate(video_namelist):
        img_namelist = os.listdir(frame_path+video_name)
        
        with open(label_path+video_name+'.txt') as f:
            labels = [i[:-1] for i in list(f)]

        hidden = model.init_hidden()
        # hidden = hidden.detach()
        running_loss = 0
        loss = 0
        # hidden_local = hidden
        for i, img in enumerate(([io.imread(frame_path+video_name+'/'+img_name) for img_name in img_namelist])):
            img_tensor = transform(img).cuda()
            resnet_out = resnet50(img_tensor.view((1,)+img_tensor.size()))
            model.zero_grad()
            # rnn_out, hidden_local = model(resnet_out.view(1, 1, -1), hidden_local)
            rnn_out, hidden = model(resnet_out.view(1, 1, -1), hidden)

            target = torch.empty(1, dtype=torch.long).fill_(float(labels[i])).cpu()
            loss += criterion(rnn_out.cpu(), target)
            # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
            
        running_loss = loss.item()
        print('[%d/%d][%d/%d] loss: %f'%(epoch+1, epoch_num, j+1, len(video_namelist), running_loss/len(img_namelist)))
        loss_history.append(running_loss/len(img_namelist))
        writer.add_scalar('loss', running_loss/len(img_namelist), step)
        step += 1

    
    torch.save(model.state_dict(), 'models/task3_LSTM_epoch%d.pth'%(epoch+1))
    np.save('task3_loss.npy', np.array(loss_history))
 
