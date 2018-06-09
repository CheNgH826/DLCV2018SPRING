from reader import readShortVideo, getVideoList
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from model import Task1Model, resnet50, Task1ModelDeep, transform
import torch.optim as optim
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter 

writer = SummaryWriter('summary/task1')

video_path = 'HW5_data/TrimmedVideos/video/train/'
label_csv_path = '/home/lilioo826/DLCV_hw5/HW5_data/TrimmedVideos/label/gt_train.csv'
feature_path = 'feature/'
label_df = pd.read_csv(label_csv_path)

classifier = Task1ModelDeep().cuda()
classifier.train()
print(classifier)

# optimizer = optim.SGD(classifier.parameters(), lr=0.1)
optimizer = optim.Adam(classifier.parameters(), lr=1e-4, betas=(0.5,0.999))
# optimizer = optim.Adamax(classifier.parameters(), lr=1e-5)
criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
epoch_num = 100
batch_size = 128
loss_history = []
for epoch in range(epoch_num):
    running_loss = 0.0
    # for idx in range(len(label_df)):
    for idx in range(0, len(label_df), batch_size):
        # video_name = label_df['Video_name'][idx]
        video_name = label_df['Video_name'][idx: idx+batch_size]
        feature_list = []
        for name in video_name: 
            feature = np.load(feature_path+name+'.npy')
            feature_mean = np.mean(feature, axis=0).flatten()
            feature_list.append(feature_mean)
        feature_list = np.array(feature_list)
        # out_mean = torch.tensor(np.mean(feature_list, axis=0)).cuda()
        out_mean = torch.tensor(feature_list).cuda()

        optimizer.zero_grad()
        classifier_out = classifier(out_mean).cpu()
        target_list = []
        for label in label_df['Action_labels'][idx: idx+batch_size]:
            target = torch.empty(1, dtype=torch.long).fill_(float(label)).cpu()
            target_list.append(target)
        loss = criterion(classifier_out, torch.cat(target_list))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (idx+1)%batch_size == 0:
            print('[Epoch %d/%d][%d/%d] loss: %f'%(epoch+1, epoch_num, idx+1, len(label_df), running_loss/(idx+1)))
    loss_history.append(running_loss)
    writer.add_scalar('loss', running_loss, epoch+1)

    if (epoch+1)%20 == 0:
        model_out_path = 'models/task1_epoch%d_deep.pth'%(epoch+1)
        torch.save({'state_dict': classifier.state_dict(), 'optimizer': optimizer.state_dict()}, model_out_path)
        os.system('python3 task1_eval.py %s'%model_out_path)

np.save('task1_loss.npy', np.array(loss_history))